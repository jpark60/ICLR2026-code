import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_task_class_ranges(nb_cl_first, nb_cl, nb_groups):
    """Return list of np.array of class IDs per task."""
    ranges = []
    # Task 1
    ranges.append(np.arange(0, nb_cl_first))
    # Later tasks
    for g in range(nb_groups):
        start = nb_cl_first + g * nb_cl
        ranges.append(np.arange(start, start + nb_cl))
    return ranges

def eval_per_task(model, X_test, y_test, nb_cl_first, nb_cl, cur_iter, 
                  batch_size=256, end_class=None):
    """
    eval acc per task
    
    Args:
        cur_iter: current task index
    Returns:
        list: [acc_task1, acc_task2, …, acc_task_cur]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    results = []

    for task_id in range(cur_iter + 1):
        if task_id == 0:
            cls_range = np.arange(0, nb_cl_first)
        else:
            start = nb_cl_first + (task_id - 1) * nb_cl
            cls_range = np.arange(start, start + nb_cl)

        mask = np.isin(y_test, cls_range)
        if mask.sum() == 0:
            results.append(np.nan)
            continue

        X_task = X_test[mask]
        y_task = y_test[mask]
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_task), 
            torch.LongTensor(y_task)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        correct = 0
        count = 0
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                # EmberTransformerPyTorch returns (logits, features)
                logits, _ = model(xb)
                logits = logits[:, :end_class]
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)

                correct += (pred == yb).sum().item()
                count += xb.size(0)

        results.append(correct / count)

    return results

def eval_global_accuracy(model, X_test, y_test, end_class, batch_size=256, device='cpu'):
    """
    global acc(task that have been learned so far)
    """
    mask_seen = y_test < end_class
    
    if mask_seen.sum() == 0:
        return 0.0
    
    X_seen = X_test[mask_seen]
    y_seen = y_test[mask_seen]
    
    # Prediction
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_seen), batch_size):
            batch_X = torch.FloatTensor(X_seen[i:i+batch_size]).to(device)
            logits, _ = model(batch_X)
            logits = logits[:, :end_class]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    all_preds = np.array(all_preds)
    accuracy = accuracy_score(y_seen, all_preds)
    return accuracy

def evaluate_model_performance(model, X_test, y_test, nb_cl_first, nb_cl, itera, 
                             end_class):
    """
    Args:
        model: model to evaluate
        X_test: test data
        y_test: test label
        nb_cl_first: number of classes in the first task
        nb_cl: number of classes per task
        itera: current iteration
        end_class: number of classes learned so far
        
    Returns:
        dict: result
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Test data preparation
    mask_seen = y_test < end_class
    X_test_seen = X_test[mask_seen]
    y_test_seen = y_test[mask_seen]

    # Task-wise accuracy calculation
    acc_task_list = eval_per_task(model, X_test, y_test,
                                  nb_cl_first, nb_cl, itera, 
                                  batch_size=256, end_class=end_class)

    # Task-wise accuracy output
    for t_id, acc in enumerate(acc_task_list, 1):
        print(f"> Task {t_id} accuracy after Iter {itera+1}: {acc:.4f}")

    # test accuracy calculation
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_seen), 
        torch.LongTensor(y_test_seen)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _ = model(x_batch)
            logits = logits[:, :end_class]
            probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(probs, dim=1)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Task {itera+1}, Overall Accuracy: {accuracy:.4f}")
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    print("[DEBUG] Unique Predictions:", dict(zip(unique_preds, pred_counts)))
    print("[DEBUG] Unique Labels:", dict(zip(unique_labels, label_counts)))

    if len(np.unique(all_labels)) > 1:
        f1_post_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        f1_post_macro = 0.0
    return {
        'task_accuracies': acc_task_list.copy(),
        'overall_accuracy': accuracy,
        'f1_macro': f1_post_macro,
        'unique_predictions': dict(zip(unique_preds, pred_counts)),
        'unique_labels': dict(zip(unique_labels, label_counts)),
    }


def print_results_summary(iteration_results, nb_groups):

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    def find_iteration_results(itera_num):

        before_result = None
        after_result = None
        
        for result in iteration_results:
            if result['iteration'] == itera_num:
                if result['phase'] == 'before_exemplar_fit':
                    before_result = result
                elif result['phase'] == 'after_exemplar_fit':
                    after_result = result
        
        return before_result, after_result
    
    # print Task 1 result
    print("\n[Task 1 result]")
    print("-" * 60)
    for result in iteration_results:
        if result['iteration'] == 0 and result['phase'] == 'before_exemplar_fit':
            task1_acc = result['task_accuracies'][0]
            task1_f1 = result['f1_macro']
            print(f"Task 1 acc: {task1_acc:.4f}")
            print(f"Task 1 F1-macro: {task1_f1:.4f}")
            break
    
    # refinement before/after comparison
    print("\n[Task-wise performance change (refinement before -> after)]")
    print("-" * 60)

    # Iterate through each iteration
    for itera_num in range(1, nb_groups + 1):  # The first iteration (0) has no refinement
        before_result, after_result = find_iteration_results(itera_num)
        if before_result and after_result:
            print(f"\n▶ Iteration {itera_num+1}:")

            # Task-wise accuracy comparison
            before_accs = before_result['task_accuracies']
            after_accs = after_result['task_accuracies']

            print(f"  [Task-wise accuracy change]")
            for task_idx in range(len(before_accs)):
                before_val = before_accs[task_idx]
                after_val = after_accs[task_idx]
                if not np.isnan(before_val) and not np.isnan(after_val):
                    improvement = after_val - before_val
                    print(f"    Task {task_idx + 1:2d}: {before_val:.4f} -> {after_val:.4f} ({improvement:+.4f})")
                else:
                    print(f"    Task {task_idx + 1:2d}: N/A")

            # overall
            before_acc = before_result['overall_accuracy']
            after_acc = after_result['overall_accuracy']
            acc_improvement = after_acc - before_acc
            print(f"\n  [Overall accuracy change]")
            print(f"    Overall: {before_acc:.4f} -> {after_acc:.4f} ({acc_improvement:+.4f})")

            # F1
            before_f1 = before_result['f1_macro']
            after_f1 = after_result['f1_macro']
            f1_improvement = after_f1 - before_f1
            print(f"\n  [F1-macro change]")
            print(f"    F1-macro: {before_f1:.4f} -> {after_f1:.4f} ({f1_improvement:+.4f})")
    
    
    print("\n" + "="*60)
    
    # summary
    print("[Final Results Summary]")
    print("-" * 60)
    
    print("\n[Overall Accuracy per Iteration]")
    
    # Task 1 (iteration 0)
    task1_result = None
    for result in iteration_results:
        if result['iteration'] == 0 and result['phase'] == 'before_exemplar_fit':
            task1_result = result
            break
    if task1_result:
        print(f"  Task 1 (Iter 1): {task1_result['overall_accuracy']:.4f}")
    
    # before/after
    for itera_num in range(1, nb_groups + 1):
        before_result, after_result = find_iteration_results(itera_num)
        if before_result:
            print(f"  Task {itera_num+1} (Iter {itera_num+1}) - Before: {before_result['overall_accuracy']:.4f}")
        if after_result:
            print(f"  Task {itera_num+1} (Iter {itera_num+1}) - After:  {after_result['overall_accuracy']:.4f}")
    
    print("="*80)


# ======================== FORGETTING SCORE ========================

def top_1_accuracy(predictions, true_labels):
    """
    Calculate the Top-1 accuracy.
    
    :param predictions: List of predicted labels
    :param true_labels: List of true labels
    :return: Top-1 accuracy
    """
    assert len(predictions) == len(true_labels), "Predictions and true_labels must have the same length"
    
    # Count the number of correct predictions
    correct_predictions = sum(p == t for p, t in zip(predictions, true_labels))
    
    # Calculate the Top-1 accuracy
    top_1_accuracy = correct_predictions / len(predictions)
    
    return top_1_accuracy


def select_samples_of_only_previous_steps(preds, labels, indices, indices_test_end_step, step):
    selected_indices = np.where(indices < indices_test_end_step[step-1])[0]
    return preds[selected_indices], labels[selected_indices]


def calculate_forgetting_score_v5(current_iter, task_accuracies_history, task_best_acc_list):
    if current_iter == 0:

        current_task_acc_list = task_accuracies_history[current_iter]
        task_best_acc_list.extend(current_task_acc_list)
        return None, current_task_acc_list
    
    current_task_acc_list = task_accuracies_history[current_iter]
    
    old_task_acc_list = current_task_acc_list[:-1]
    previous_best_accs = task_best_acc_list[:len(old_task_acc_list)]
    
    forgetting_score = np.mean(np.array(previous_best_accs) - np.array(old_task_acc_list))
    
    for i in range(len(old_task_acc_list)):
        task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])

    task_best_acc_list.append(current_task_acc_list[-1])
    
    return forgetting_score, current_task_acc_list


def calculate_forgetting_score(step, preds_mapped, labels_mapped, task_best_acc_list, 
                             incremental_nbr_new_classes, val_init_samp_indices_list, 
                             indices_test_end_step):
    
    predictions = np.array(preds_mapped)
    true_labels = np.array(labels_mapped)
    init_samp_indices = np.array(val_init_samp_indices_list)

    if step > 0:
        predictions, true_labels = select_samples_of_only_previous_steps(
            predictions, true_labels, init_samp_indices, indices_test_end_step, step)

    old_task_acc_list = []
    for i in range(step+1):
        step_class_list = range(incremental_nbr_new_classes[i], incremental_nbr_new_classes[i+1])
        
        step_class_idxs = []
        for c in step_class_list:
            idxs = np.where(true_labels == c)[0].tolist()
            step_class_idxs += idxs
        step_class_idxs = np.array(step_class_idxs)
        
        if len(step_class_idxs):
            i_labels = true_labels[step_class_idxs]
            i_logits = predictions[step_class_idxs]
        else:
            i_labels = true_labels
            i_logits = predictions

        i_acc = top_1_accuracy(i_logits, i_labels)
        if i == step:
            curren_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)
            
    if step > 0:
        forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None

    task_best_acc_list.append(curren_step_acc)

    return forgetting, old_task_acc_list + [curren_step_acc]
