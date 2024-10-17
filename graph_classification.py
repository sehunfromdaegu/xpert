import os
import random
import numpy as np
import argparse

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from timm.scheduler import CosineLRScheduler

from utils import convert_to_epd_list, rotate_epds, pixelization, train_one_epoch, test
from load_models import load_model
from gdeep.data.datasets.persistence_diagrams_from_graphs_builder import PersistenceDiagramFromGraphBuilder
from gdeep.data.datasets import PersistenceDiagramFromFiles

def labels_preprocess(labels, dataname):
    """Preprocess labels based on dataset name."""
    if dataname in ['IMDB-MULTI', 'PROTEINS']:
        labels = labels - 1
    if any(name in dataname for name in ['MUTAG', 'COX2', 'DHFR']):
        labels = 0.5 * labels + 0.5    
    return labels

def main(args):
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract arguments
    dataname = args.dataname
    model = args.model
    grid_size = args.grid_size
    patch_size = args.patch_size
    embed_dim = args.embed_dim
    depth = args.depth
    epochs = args.epochs
    lr = args.lr
    warmup_t = args.warmup_t
    batch_size = args.batch_size
    n_splits = args.n_splits

    print(f"Dataset: {dataname}, Model: {model}, Patch Size: {patch_size}, Embed Dim: {embed_dim}, Depth: {depth}")

    # Load the dataset
    dataset = TUDataset(root='./data/GraphDatasets/', name=dataname)
    num_classes = dataset.num_classes

    # Initialize tensor for pixelized persistence diagrams
    ppd = torch.zeros((len(dataset), 4, grid_size, grid_size), dtype=torch.float32)

    # Create persistence diagrams from graphs
    diffusion_parameter = 1.0
    pd_creator = PersistenceDiagramFromGraphBuilder(dataname, diffusion_parameter=diffusion_parameter, root='./data')
    pd_creator.create()

    # Load persistence diagrams
    pd_ds = PersistenceDiagramFromFiles(
        os.path.join('./data', f"{dataname}_{diffusion_parameter}_extended_persistence")
    )

    # Preprocess labels
    labels = [pd_ds[i][1] for i in range(len(pd_ds))]
    labels = np.array(labels)
    labels = labels_preprocess(labels, dataname)
    print(f'{dataname} labels: {np.unique(labels)}')

    # Convert and rotate persistence diagrams
    epds, _ = convert_to_epd_list(pd_ds)
    repds = rotate_epds(epds)  # List of rotated persistence diagrams

    # Pixelize persistence diagrams
    for i in range(len(repds)):
        ppd[i] = pixelization(repds[i], grid_size=grid_size, device='cpu')

    # Verify that graph labels match persistence diagram labels
    graph_labels = [dataset[i].y.item() for i in range(len(dataset))]
    num_same_labels = (np.array(graph_labels) == labels).sum()
    sanity = (num_same_labels == len(dataset))
    print(f"Graph labels are the same as the labels in the persistence diagram dataset: {sanity}")

    if not sanity:
        print("Warning: Graph labels do not match persistence diagram labels.")

    # Prepare data list
    data_list = []
    for idx, data in enumerate(dataset):
        data.node_feat = torch.ones((data.num_nodes, 1), dtype=torch.float32)
        data.ppd = ppd[idx]  # ppd[i].shape = (4, grid_size, grid_size)
        data_list.append(data)

    # Stratified K-Fold cross-validation
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    best_test_acc_list = []

    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(skfold.split(data_list, labels)):
        print(f"Starting Fold {fold + 1}/{n_splits}")

        # Load model
        model = load_model(
            model, device, num_classes, grid_size, patch_size,
            depth=depth, embed_dim=embed_dim
        )

        # Data loaders
        train_loader = DataLoader(Subset(data_list, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(data_list, test_idx), batch_size=batch_size)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epochs,
            cycle_mul=1,
            lr_min=0.05 * lr,
            cycle_decay=1.0,
            warmup_lr_init=0.05 * lr,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True
        )

        best_test_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0  # Counter for epochs with no improvement

        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler=scheduler)

            # Validation
            test_loss, test_acc = test(model, test_loader, criterion, device)

            # Early stopping logic
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        print(f'Fold {fold + 1}/{n_splits} - Best Test Accuracy: {best_test_acc:.3f}')
        best_test_acc_list.append(best_test_acc)

    # Final results
    avg_acc = np.mean(best_test_acc_list)
    std_acc = np.std(best_test_acc_list)
    print(f'Average Best Test Accuracy over {n_splits} folds: {avg_acc:.3f} ± {std_acc:.3f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on dataset with persistence diagrams")
    parser.add_argument('--dataname', type=str, default='PROTEINS', help="Dataset name") # Choose from ['IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'PROTEINS', 'COX2', 'DHFR'] 
    parser.add_argument('--model', type=str, default='xpert', help="Model name") # Choose from ['xpert', 'gin', 'gin_assisted_concat', 'gin_assisted_sum']
    parser.add_argument('--grid_size', type=int, default=50, help="Grid size for persistence diagram")
    parser.add_argument('--patch_size', type=int, default=5, help="Patch size for pixelization")
    parser.add_argument('--embed_dim', type=int, default=192, help="Embedding dimension")
    parser.add_argument('--depth', type=int, default=5, help="Depth of the model")
    parser.add_argument('--epochs', type=int, default=300, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--warmup_t', type=int, default=50, help="Warmup steps for the scheduler")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--n_splits', type=int, default=10, help="Number of splits for cross-validation")
    args = parser.parse_args()
    main(args)
