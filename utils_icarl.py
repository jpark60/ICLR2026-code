import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from utils_transformer import EmberTransformerPyTorch
from utils_data import CustomDataset, read_data3
import config

def prepare_networks(total_classes, nb_cl_first, nb_cl, nb_groups, itera=None, save_path=None, device='cpu'):

    model_train = EmberTransformerPyTorch(**config.model_config).to(device)

    model_test = EmberTransformerPyTorch(**config.model_config).to(device)

    if itera is not None and itera > 0 and save_path is not None:
        try:
            weights_path = f'{save_path}model-iteration{nb_cl}-{itera-1}.pt'
            model_train.load_state_dict(torch.load(weights_path, map_location=device))
            model_test.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded previous weights from {weights_path}")
        except:
            print("No previous weights found. Starting with fresh initialization.")
            
    return model_train, model_test


def reading_data_and_preparing_network(idx_iter, itera, batch_size, x_path, y_path, mixing, nb_groups, nb_cl, nb_cl_first, save_path, device):

    x_train = np.load(x_path)
    y_train = np.load(y_path)

    from utils_data import read_data3
    x_train_task, y_train_task = read_data3(x_train, y_train, mixing, idx_iter)

    dataset = CustomDataset(x_train_task, y_train_task)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_classes = nb_cl_first + (nb_groups * nb_cl)
    model = EmberTransformerPyTorch(**config.model_config).to(device)

    weights_path = f'{save_path}model-iteration{nb_cl}-{itera}.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    return data_loader, model


def load_class_in_feature_space(idx_iter, batch_size, dataset, model, mixing, device):

    model.eval()
    label_dico = []
    Dtot = []
    
    total_batches = int(np.ceil(len(idx_iter)/batch_size))
    
    for i, (x_batch, l) in enumerate(dataset):
        if i >= total_batches:
            break
            
        x_batch = x_batch.to(device)
        with torch.no_grad():
            feat_map_tmp = model.feature_extract(x_batch)
            normalized_prototypes = F.normalize(feat_map_tmp.T, p=2, dim=0)
            
            Dtot.append(normalized_prototypes.cpu().numpy().astype(np.float32))
            label_dico.extend(l.numpy())
    
    if Dtot:
        Dtot = np.concatenate(Dtot, axis=1)  # (n_features, n_samples)
    else:
        Dtot = np.array([]).reshape(0, 0)
    label_dico = np.array(label_dico)
    
    print(f"Dtot shape: {Dtot.shape}, label_dico shape: {label_dico.shape}")
    
    return Dtot, label_dico

