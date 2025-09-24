import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import gc

import config
import utils_exemplar
import utils_icarl
import utils_data
import utils_eval
from utils_transformer import EmberTransformerPyTorch

# config
device = config.device
batch_size = config.batch_size
nb_cl_first = config.nb_cl_first
nb_cl = config.nb_cl
nb_groups = config.nb_groups
nb_total = config.nb_total
epochs = config.epochs
lr = config.lr
wght_decay = config.wght_decay
use_weight_decay_in_exemplar = config.use_weight_decay_in_exemplar
lr_patience = config.lr_patience
stop_patience = config.stop_patience
stop_floor_ep = config.stop_floor_ep
factor = config.factor
min_lr = config.min_lr
nb_cluster = config.nb_cluster

# path
data_path = config.data_path
x_path = config.x_path
y_path = config.y_path
x_path_valid = config.x_path_valid
y_path_valid = config.y_path_valid
save_path = config.save_path

### Initialization of some variables ###
loss_batch = []
files_protoset = []
accuracy_all = []
### Save Results ###
task_accuracy_all = []
iteration_results = []
forgetting_scores = []
task_best_acc_list = []

total_classes = nb_cl_first + (nb_groups * nb_cl)  # 22 + (4 * 5) = 42
for _ in range(total_classes):
    files_protoset.append([])

### Random mixing ###
print("Mixing the classes and putting them in batches of classes...")
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

# GPU deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

order = np.arange(total_classes)
mixing = np.arange(total_classes)
np.random.shuffle(mixing)

### Preparing the files per group of classes ###
print("Creating a training set ...") 
files_train = utils_data.prepare_files(x_path, y_path, mixing, order, nb_groups, nb_cl, nb_cl_first) 
files_valid = utils_data.prepare_files(x_path_valid, y_path_valid, mixing, order, nb_groups, nb_cl, nb_cl_first) 

### Save the mixing and order ###
with open(f"{nb_cl}mixing.pickle", 'wb') as fp:
    pickle.dump(mixing, fp)

with open(f"{nb_cl}settings_mlp.pickle", 'wb') as fp:
    pickle.dump(order, fp)
    pickle.dump(files_train, fp)

##### ------------- Main Algorithm START -------------#####
for itera in range(nb_groups + 1):
    print(f'Batch of classes number {itera+1} arrives ...')
    
    if itera == 0:
        cur_nb_cl = nb_cl_first
        idx_iter = files_train[itera]
    else:
        cur_nb_cl = nb_cl
        idx_iter = files_train[itera][:]
        
        total_cl_now = nb_cl_first + ((itera-1) * nb_cl)
        nb_protos_cl = int(np.ceil(nb_total * 1.0 / total_cl_now))

        for i in range(nb_cl_first + (itera-1)*nb_cl): 
            tmp_var = files_protoset[i]
            selected_exemplars = tmp_var[0:min(len(tmp_var),nb_protos_cl)]
            idx_iter += selected_exemplars

    print(f'Task {itera + 1}: Training {cur_nb_cl} classes...') 

    X_train, y_train = utils_data.read_data(x_path, y_path, mixing, idx_iter)
    X_val, y_val = utils_data.read_data(x_path_valid, y_path_valid, mixing, files_valid[itera])
    
    # DataLoader
    train_dataset = utils_data.CustomDataset(X_train, y_train)
    val_dataset = utils_data.CustomDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle= True, num_workers=4)

    # model initialize/load, optimizer setting
    if itera == 0: 
        model_train, model_test = utils_icarl.prepare_networks(
            total_classes, nb_cl_first, nb_cl, nb_groups, itera=itera, save_path=save_path, device=device
        )
        optimizer = optim.Adam(model_train.parameters(), lr=lr, weight_decay=wght_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        # ReduceLROnPlateau scheduler (only first task)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=lr_patience, min_lr=min_lr
        )
        def train_step(model, x, y, optimizer, loss_fn):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            return loss
    else:
        model_train, model_test = utils_icarl.prepare_networks(
            total_classes, nb_cl_first, nb_cl, nb_groups, itera=itera, save_path=save_path, device=device
        )
        optimizer = optim.Adam(model_train.parameters(), lr=lr, weight_decay=wght_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        def train_step(model, x, y, optimizer, loss_fn):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            return loss

    current_epoch = 0
    stopped_early = False
    best_val_loss = float('inf')
    early_stop_patience_counter = 0
    best_weights = None
    lr_patience_counter = 0
    
    ### Training loop ###
    for epoch in range(epochs):
        epoch_losses = []

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss_value = train_step(model_train, x_batch, y_batch, optimizer, loss_fn)
            epoch_losses.append(loss_value.item())

            # Print average loss
            if (step + 1) % 50 == 0:
                print('\r', epoch, 'epoch', step, 'batch', f'Average loss: {np.mean(epoch_losses[-100:]):.6f}', end='')

        # Validation accuracy/loss
        model_train.eval()
        val_losses = []
        correct = 0
        total = 0
        debug_predictions = []
        debug_labels = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits, _ = model_train(x_val)
                val_loss = loss_fn(logits, y_val)
                val_losses.append(val_loss.item())
                end_class = nb_cl_first if itera == 0 else (nb_cl_first + nb_cl * itera)
                logits_limited = logits[:, :end_class]
                _, predicted = torch.max(logits_limited.data, 1)
                
                # debugging info
                if len(debug_predictions) == 0:
                    debug_predictions.extend(predicted.cpu().numpy()[:10])
                    debug_labels.extend(y_val.cpu().numpy()[:10])
                
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        val_acc = correct / total if total > 0 else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        improved = avg_val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = avg_val_loss
            best_weights = model_train.state_dict().copy()
            early_stop_patience_counter = 0
        else:
            early_stop_patience_counter += 1
        # ReduceLROnPlateau scheduler step
        if itera == 0:
            scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f", Val_Loss: {avg_val_loss:.4f}, Val_Acc: {val_acc:.4f}, LR: {current_lr:.6e}")
        # no early stopping in the first task
        if itera > 0:
            if early_stop_patience_counter >= stop_patience and epoch >= stop_floor_ep:
                stopped_early = True
                current_epoch = epoch + 1 
                print(f"[EarlyStop] epoch {current_epoch}, best_val_loss {best_val_loss:.4f}")
                model_train.load_state_dict(best_weights)
                break

    print(f"[Iter {itera}] training completed")

    if not stopped_early:
        current_epoch = epochs
    print(f"[Iter {itera}] used_train_epochs={current_epoch}")

    exemplar_epochs = utils_exemplar.compute_exemplar_epochs(current_epoch)
    
    print(f"[Iter {itera}] exemplar_epochs={exemplar_epochs}")

    # Save the model
    torch.save(model_train.state_dict(), save_path + f'model-iteration{nb_cl}-{itera}.pt')
    
    # copy model (before refinement)
    model_before = EmberTransformerPyTorch(**config.model_config).to(device)
    model_before.load_state_dict(model_train.state_dict())
    
    X_test = np.load(data_path+'/X_test.npy')
    y_test = np.load(data_path+'/y_test.npy')
    y_test = np.array([mixing[label] for label in y_test])
    end_class = nb_cl_first if itera == 0 else (nb_cl_first + nb_cl * itera)

    # eval (before refinement)
    eval_result = utils_eval.evaluate_model_performance(
        model_before, X_test, y_test, nb_cl_first, nb_cl, itera, 
        end_class
    )

    # save (before refinement))
    accuracy_all.append(eval_result['overall_accuracy'])
    task_accuracy_all.append(eval_result['task_accuracies'].copy())
    
    iteration_result = {
        'iteration': itera,
        'phase': 'before_exemplar_fit',
        **eval_result
    }
    iteration_results.append(iteration_result)
 
    # Exemplars management part
    total_cl_now = nb_cl_first + (itera * nb_cl)
    nb_protos_cl = int(np.ceil(nb_total * 1.0 / total_cl_now))
    print(f"\n[DEBUG] Exemplar Management - Iter {itera}:")
    print(f"  total_cl_now: {total_cl_now}, nb_total: {nb_total}, nb_protos_cl: {nb_protos_cl}")
    idx_iter = files_train[itera]
    
    dataset, model = utils_icarl.reading_data_and_preparing_network(
        idx_iter, itera, batch_size, x_path, y_path, mixing, nb_groups, nb_cl, nb_cl_first, save_path, device
    )

    # Load the training samples of the current batch of classes in the feature space
    Dtot, label_dico = utils_icarl.load_class_in_feature_space(
        idx_iter, batch_size, dataset, model, mixing, device
    )
    print("After function call - Dtot has NaN:", np.isnan(Dtot).any())

    # kmeans exemplar selection
    print('Exemplars selection starting ...')
    if itera == 0:
        start_idx = 0
        end_idx = nb_cl_first
    else:
        start_idx = nb_cl_first + (itera-1)*nb_cl
        end_idx = nb_cl_first + itera*nb_cl

    for iter_dico in range(end_idx - start_idx):
        current_cl = start_idx + iter_dico
        ind_cl = np.where(label_dico == order[current_cl])[0]
        D = Dtot[:, ind_cl]

        if len(ind_cl) == 0:
            print(f"Warning: No samples found for class {current_cl} in current task data")
            continue

        idx_iter_arr = np.array(idx_iter)
        files_iter = idx_iter_arr[ind_cl]

        assert len(D[0]) == len(files_iter), f"D vs files_iter length mismatch: D={len(D[0])}, files_iter={len(files_iter)}"

        print(f"\n[DEBUG] Class {current_cl} K-means Exemplar Selection:")

        #### choose kmeans or random or c-mean
        # kmeans exemplar selection 
        selected_exemplars_files = utils_exemplar.kmeans_exemplar_selection(nb_cluster, D, files_iter, nb_protos_cl)
        # C-mean/random exemplar selection 
        # selected_exemplars_files = utils_exemplar.mean_exemplar_selection(D, files_iter, nb_protos_cl, method='random')

        for exemplar_file in selected_exemplars_files:
            if exemplar_file not in files_protoset[current_cl]:
                files_protoset[current_cl].append(exemplar_file)

        print(f"  - Total exemplars for class {current_cl}: {len(files_protoset[current_cl])}/{nb_protos_cl}")

        gc.collect()
            
    with open(f"{nb_cl}_files_protoset.pickle", "wb") as fp:
        pickle.dump(files_protoset, fp)

    if itera > 0:
        start_idx = nb_cl_first + (itera-1)*nb_cl
        
        all_exemplar_indices = []

        ex_lr = lr * 0.1
        ex_opt = optim.Adam(
            model_train.parameters(), 
            lr=ex_lr,
            weight_decay=wght_decay if use_weight_decay_in_exemplar else 0.0
        )

        for class_id, exemplar_list in enumerate(files_protoset):
            if len(exemplar_list) > 0:
                if class_id < (nb_cl_first + itera * nb_cl):
                    selected_exemplars = exemplar_list[:min(len(exemplar_list), nb_protos_cl)]
                    all_exemplar_indices.extend(selected_exemplars)
                    print(f"Class {class_id}: Using {len(selected_exemplars)}/{len(exemplar_list)} exemplars for retraining")

        # no exemplar -> skip
        if len(all_exemplar_indices) > 0:
            X_ex, y_ex = utils_data.read_data(x_path, y_path, mixing, all_exemplar_indices)

            # create Exemplar dataset
            ex_dataset = utils_data.CustomDataset(X_ex, y_ex)
            ex_loader = DataLoader(ex_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            # exemplar-only fine-tuning loop
            utils_exemplar.run_exemplar_fine_tuning(
                model_train, model_before, model_test, ex_loader,
                exemplar_epochs, loss_fn, ex_opt, start_idx, device, a=4.0, b=1.0
            )
        else:
            print('\n[EXEMPLAR-FIT] (no exemplars yet skipping)')

        # save weight, eval accuracy
        torch.save(model_train.state_dict(), save_path + f'model-iteration{nb_cl}-{itera}.pt')

        # overfitting eval
        print("[Exemplar-fit] done.")
        model_test.load_state_dict(model_train.state_dict()) 

        # eval (after refinement)
        eval_result_after = utils_eval.evaluate_model_performance(
            model_test, X_test, y_test, nb_cl_first, nb_cl, itera, 
            end_class
        )

        iteration_result_after = {
            'iteration': itera,
            'phase': 'after_exemplar_fit',
            **eval_result_after
        }
        iteration_results.append(iteration_result_after)
        
        # Forgetting Score & update
        task_accuracy_all[-1] = eval_result_after['task_accuracies'].copy()
        
        forgetting_score, current_task_acc_list = utils_eval.calculate_forgetting_score_v5(
            itera, task_accuracy_all, task_best_acc_list
        )
        
        if forgetting_score is not None:
            forgetting_scores.append(forgetting_score)
            print(f"\n[FORGETTING SCORE] Iteration {itera} (after refinement): {forgetting_score:.4f}")
            print(f"[FORGETTING SCORE] Average forgetting so far: {np.mean(forgetting_scores):.4f}")
        else:
            print(f"\n[FORGETTING SCORE] Iteration {itera}: N/A (first task)")

    else:
        # itera == 0
        forgetting_score, current_task_acc_list = utils_eval.calculate_forgetting_score_v5(
            itera, task_accuracy_all, task_best_acc_list
        )
        
        if forgetting_score is not None:
            forgetting_scores.append(forgetting_score)
            print(f"\n[FORGETTING SCORE] Iteration {itera}: {forgetting_score:.4f}")
            print(f"[FORGETTING SCORE] Average forgetting so far: {np.mean(forgetting_scores):.4f}")
        else:
            print(f"\n[FORGETTING SCORE] Iteration {itera}: N/A (first task)")

    torch.cuda.empty_cache()
    gc.collect()

# Final results summary
utils_eval.print_results_summary(iteration_results, nb_groups)

# Forgetting Score summary
print("\n" + "="*60)
print("FORGETTING SCORE SUMMARY")
print("="*60)
if forgetting_scores:
    print(f"Forgetting scores per iteration: {[f'{f:.4f}' for f in forgetting_scores]}")
    print(f"Average forgetting score: {np.mean(forgetting_scores):.4f}")
    print(f"Final task best accuracies: {[f'{acc:.4f}' for acc in task_best_acc_list]}")
else:
    print("No forgetting scores calculated (only one task)")
print("="*60)
