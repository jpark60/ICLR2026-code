import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
import gc

from utils_data import read_data3, CustomDataset
from utils_transformer import EmberTransformerPyTorch
import config


def exemplar_epochs_from_used(used_train, scale=1/3, floor=8, cap=15):
    ep = int(round(used_train * scale))
    ep = max(floor, ep)
    if cap is not None:
        ep = min(cap, ep)
    return ep


def compute_exemplar_epochs(current_epoch):
    return exemplar_epochs_from_used(
        current_epoch, 
        scale=config.exemplar_scale, 
        floor=config.exemplar_floor, 
        cap=config.exemplar_cap
    )


def kmeans_exemplar_selection(nb_cluster, D, files_iter, nb_protos_cl):
    n_samples = len(files_iter)
    
    if n_samples <= nb_protos_cl:
        return files_iter.tolist()
    
    # cluster define
    n_clusters = min(nb_cluster, n_samples)
    # n_clusters = min(nb_protos_cl, n_samples)
    
    if n_clusters <= 1:
        # 
        np.random.seed(config.SEED)
        selected_indices = np.random.choice(n_samples, nb_protos_cl, replace=False)
        return [files_iter[i] for i in selected_indices]
    
    # K-means clustering
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, max_iter=100, random_state=config.SEED, n_init=10)
    kmeans.fit(D.T)  # (N, feature_dim)
    
    # cluster size
    cluster_sizes = np.bincount(kmeans.labels_, minlength=n_clusters)
    total_samples = sum(cluster_sizes)

    # Allocate exemplars proportional to cluster size
    samples_per_cluster = {}
    total_allocated = 0
    
    for cluster_id in range(n_clusters):
        proportion = cluster_sizes[cluster_id] / total_samples
        n_samples_cluster = int(proportion * nb_protos_cl) 
        samples_per_cluster[cluster_id] = n_samples_cluster
        total_allocated += n_samples_cluster

    # Distribute remaining samples (in order of cluster size)
    remaining = nb_protos_cl - total_allocated
    if remaining > 0:
        sorted_clusters = np.argsort(cluster_sizes)[::-1]
        for i in range(remaining):
            cluster_id = sorted_clusters[i % n_clusters]
            samples_per_cluster[cluster_id] += 1

    # Select exemplars
    selected_exemplars = []
    for cluster_idx, n_samples in samples_per_cluster.items():
        if n_samples == 0:
            continue
            
        cluster_samples_idx = np.where(kmeans.labels_ == cluster_idx)[0]
        
        if len(cluster_samples_idx) == 0:
            continue
        
        cluster_samples = D.T[cluster_samples_idx]
        distances = np.linalg.norm(cluster_samples - kmeans.cluster_centers_[cluster_idx], axis=1)
        
        sorted_idx = cluster_samples_idx[np.argsort(distances)]
        n_select = min(n_samples, len(cluster_samples_idx))
        selected_exemplars.extend(sorted_idx[:n_select])
    
    selected_files = [files_iter[idx] for idx in selected_exemplars]
    
    del kmeans
    gc.collect()
    
    return selected_files


def create_distillation_train_step(model_train, model_before, model_test, criterion, optimizer, start_idx, device, a=4.0, b=1.0):

    def train_step_ex(x, y):
        model_train.train()
        optimizer.zero_grad()
        
        # Forward passes
        logits, _ = model_train(x)
        
        with torch.no_grad():
            logits_before, _ = model_before(x)
            logits_prev, _ = model_test(x)
        
        # Classification loss
        ce_loss = criterion(logits, y)
        
        # Get labels  
        y_labels = y
        past_mask = y_labels < start_idx
        current_mask = y_labels >= start_idx
        
        total_loss = ce_loss
        
        # Previous task distillation 
        if torch.sum(past_mask) > 0:
            d_p_each = torch.mean(torch.square(logits_prev.detach() - logits), dim=1)
            distill_past = torch.mean(d_p_each[past_mask])
            total_loss += a * distill_past
        
        # Current task distillation
        if torch.sum(current_mask) > 0:
            train_each = torch.mean(torch.square(logits_before.detach() - logits), dim=1)
            distill_cur = torch.mean(train_each[current_mask])
            total_loss += b * distill_cur
        
        total_loss.backward()
        optimizer.step()
        
        return total_loss
    
    return train_step_ex


def run_exemplar_fine_tuning(model_train, model_before, model_test, ex_loader, exemplar_epochs, loss_fn, optimizer, start_idx, device, a=4.0, b=1.0):

    print(f'\n[Refinement] Starting refinement for {exemplar_epochs} epochs...')
    model_train.train()
    model_before.eval()
    model_test.eval()

    train_step_ex = create_distillation_train_step(
        model_train, model_before, model_test, loss_fn, optimizer, start_idx, device, a=a, b=b
    )
    for epoch in range(exemplar_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for inputs, labels in ex_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = train_step_ex(inputs, labels)
            epoch_loss += loss.item() if hasattr(loss, 'item') else float(loss)
            num_batches += 1
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if (epoch + 1) % 5   == 0 or epoch == 0:
            print(f'[Refinement] Epoch {epoch+1}/{exemplar_epochs}, Loss: {avg_loss:.6f}')
    print(f'[Refinement] Completed {exemplar_epochs} epochs')

def mean_exemplar_selection(D, files_iter, nb_protos_cl, method='mean'):
    # D: (feature_dim, n_samples)
    
    n_samples = D.shape[1]
    selected_exemplars = []
    
    if method == 'random':
        np.random.seed(config.SEED)

        all_indices = np.arange(n_samples)
        np.random.shuffle(all_indices)
        selected_exemplars = all_indices[:min(nb_protos_cl, n_samples)].tolist()
    elif method == 'mean':
        mu = np.mean(D, axis=1)  # class mean
        w_t = mu.copy()
        step_t = 0
        
        # filling 
        while len(selected_exemplars) < nb_protos_cl and step_t < n_samples * 2:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            w_t = w_t + mu - D[:, ind_max]
            step_t += 1
            
            # check duplication
            if ind_max not in selected_exemplars:
                selected_exemplars.append(ind_max)
        
        # fill if not enough exemplars
        if len(selected_exemplars) < nb_protos_cl:
            remaining_indices = [i for i in range(n_samples) if i not in selected_exemplars]
            needed = nb_protos_cl - len(selected_exemplars)
            if len(remaining_indices) > 0:
                additional = np.random.choice(remaining_indices, min(needed, len(remaining_indices)), replace=False)
                selected_exemplars.extend(additional.tolist())
    
    else:  # fallback random selection
        np.random.seed(config.SEED)
        all_indices = np.arange(n_samples)
        np.random.shuffle(all_indices)
        selected_exemplars = all_indices[:min(nb_protos_cl, n_samples)].tolist()

    # Return file indices
    selected_files = [files_iter[idx] for idx in selected_exemplars]
    return selected_files