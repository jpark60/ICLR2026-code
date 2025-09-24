import os
import torch
import argparse

def parse_arguments():
    """Parse command line arguments for continual learning experiment"""
    parser = argparse.ArgumentParser(description='Continual Learning with iCaRL')
    
    # Dataset Selection
    parser.add_argument('--dataset', type=str, choices=['cic', 'iot'], default='cic',
                        help='Dataset to use: cic (CICAndMal2017) or iot (IoT23) (default: cic)')

    # GPU and System Settings
    parser.add_argument('--gpu', type=str, default='3',
                        help='GPU device to use (default: 3)')
    parser.add_argument('--seed', type=int, default=93,
                        help='Random seed (default: 93)')
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Weight decay (default: 0.00001)')
    parser.add_argument('--use_weight_decay_in_exemplar', action='store_true',
                        help='Use weight decay in exemplar training')
    
    # Continual Learning Settings
    parser.add_argument('--nb_cl_first', type=int, default=None,
                        help='Number of classes in first task')
    parser.add_argument('--nb_cl', type=int, default=None,
                        help='Number of classes per task')
    parser.add_argument('--nb_groups', type=int, default=None,
                        help='total task - 1')
    parser.add_argument('--nb_total', type=int, default=None,
                        help='Total exemplar buffer size')
    
    # Learning Rate Scheduler
    parser.add_argument('--lr_patience', type=int, default=2,
                        help='LR scheduler patience (default: 2)')
    parser.add_argument('--stop_patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR reduction factor (default: 0.5)')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Minimum learning rate (default: 1e-7)')
    parser.add_argument('--stop_floor_ep', type=int, default=10,
                        help='Minimum epoch for early stopping (default: 10)')
    
    # Exemplar Settings
    parser.add_argument('--exemplar_scale', type=float, default=1/3,
                        help='Exemplar epochs scaling factor (default: 1/3)')
    parser.add_argument('--exemplar_floor', type=int, default=None,
                        help='Minimum exemplar epochs')
    parser.add_argument('--exemplar_cap', type=int, default=None,
                        help='Maximum exemplar epochs')
    parser.add_argument('--nb_cluster', type=int, default=None,    
                        help='K-means cluster size for exemplar selection')
    
    # Data Paths
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--save_path', type=str, default='./',
                        help='Path to save models (default: ./)')
    
    # Model Architecture
    parser.add_argument('--in_feats', type=int, default=None,
                        help='Input features')
    parser.add_argument('--hidden', type=int, default=None,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of transformer layers')
    parser.add_argument('--nhead', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    parser.add_argument('--use_cls_token', type=bool, default=None,
                        help='Use CLS token in transformer')
    
    return parser.parse_args()

def get_dataset_config(dataset_name):
    """dataset-specific configuration"""
    if dataset_name == 'cic':
        return {
            'nb_cl_first': 22,
            'nb_cl': 5,
            'nb_groups': 4,
            'nb_total': 33000,
            'data_path': '/scratch/Malware/CICAndMal/processed_data/',
            'in_feats': 85,
            'epochs': 50,
            # Model architecture for cic dataset
            'hidden': 384,
            'num_layers': 6,
            'nhead': 8,
            'dropout': 0.1,
            'use_cls_token': True,
            # Exemplar settings for cic dataset
            'exemplar_floor': 8,
            'exemplar_cap': 10000,
            'nb_cluster': 800,
        }
    elif dataset_name == 'iot':
        return {
            'nb_cl_first': 5,
            'nb_cl': 1,
            'nb_groups': 4,
            'nb_total': 10000,
            'data_path': '/scratch/Malware/iot23/data/',
            'in_feats': 23,
            'epochs': 40,
            # Model architecture for IoT dataset
            'hidden': 16,
            'num_layers': 1,
            'nhead': 2,
            'dropout': 0.2,
            'use_cls_token': True,
            # Exemplar settings for iot dataset
            'exemplar_floor': 20,
            'exemplar_cap': 30,
            'nb_cluster': 600,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

args = parse_arguments()

dataset_config = get_dataset_config(args.dataset)

# Override arguments with dataset-specific values if not explicitly set
for key, value in dataset_config.items():
    if getattr(args, key) is None:
        setattr(args, key, value)

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration from arguments
SEED = args.seed
batch_size = args.batch_size
nb_cl_first = args.nb_cl_first
nb_cl = args.nb_cl
nb_groups = args.nb_groups
nb_groups_first = 1
nb_total = args.nb_total
epochs = args.epochs
lr = args.lr
wght_decay = args.weight_decay
use_weight_decay_in_exemplar = args.use_weight_decay_in_exemplar
total_classes = nb_cl_first + (nb_groups * nb_cl)

# Learning Rate Scheduler
lr_patience = args.lr_patience
stop_patience = args.stop_patience
stop_floor_ep = args.stop_floor_ep
factor = args.lr_factor
min_lr = args.min_lr

# Exemplar
exemplar_scale = args.exemplar_scale
exemplar_floor = args.exemplar_floor
exemplar_cap = args.exemplar_cap
nb_cluster = args.nb_cluster

# path
data_path = args.data_path
x_path = f"{data_path}/X_train.npy"
y_path = f"{data_path}/y_train.npy"
x_path_valid = f"{data_path}/X_valid.npy"
y_path_valid = f"{data_path}/y_valid.npy"
save_path = args.save_path

# model
model_config = {
    'in_feats': args.in_feats,
    'num_classes': total_classes,
    'hidden': args.hidden,
    'mlp_hidden': args.hidden * 3,
    'num_layers': args.num_layers,
    'nhead': args.nhead,
    'dropout': args.dropout,
    'use_cls_token': args.use_cls_token
}

# print configuration
print(f"\n{'='*60}")
print(f"CONFIGURATION - Dataset: {args.dataset.upper()}")
print(f"{'='*60}")
print(f"Data Path: {data_path}")
print(f"Exemplars: Total Buffer={nb_total}")
print(f"Model: Features={args.in_feats}, Hidden={args.hidden}, Layers={args.num_layers}")
print(f"Training: Epochs={epochs}, Batch Size={batch_size}, LR={lr}")
print(f"FIT epoch floor={exemplar_floor}, stop_floor_ep={stop_floor_ep}")
print(f"{'='*60}\n")

