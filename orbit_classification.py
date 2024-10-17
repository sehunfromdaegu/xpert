import time
# Include necessary general imports
import os
from typing import Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import copy
# Torch imports
from orbit_utils import *
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Gdeep imports 

from gdeep.data.datasets import OrbitsGenerator
import numpy as np

patience = 50
samples_per_class = 1000
num_repeat = 5
patch_size = 5
embed_dim = 192
depth = 5

for patch_size in [5]:
    best_test_accs = []
    print(f'depth: {depth}, embed_dim: {embed_dim} patch_size: {patch_size}, samples_per_class: {samples_per_class}')
    for i in range(num_repeat):
        model_name = 'xpert'

        config = {
            'bs': 64,
            'embed_dim': embed_dim,
            'depth': depth,
            'num_heads': 8,
            'grid_size': 50,
            'patch_size': patch_size,
        }

        hyper_config = {
            'epochs': 300,
            'lr': 0.0001,
            'warmup_t': 50,
        }

        # Generate a configuration file with the parameters of the desired dataset
        @dataclass
        class Orbit5kConfig():
            batch_size_train: int = 8
            num_orbits_per_class: int = samples_per_class
            validation_percentage: float = 0.
            test_percentage: float = 0.3
            num_jobs: int = 8
            dynamical_system: str = "classical_convention"
            homology_dimensions: Tuple[int, int] = (0, 1)  # type: ignore
            dtype: str = "float32"
            arbitrary_precision: bool = False

        config_data = Orbit5kConfig()
        valid_dgms = False


        while not valid_dgms:
            og = OrbitsGenerator(
                num_orbits_per_class=config_data.num_orbits_per_class,
                homology_dimensions=config_data.homology_dimensions,
                validation_percentage=config_data.validation_percentage,
                test_percentage=config_data.test_percentage,
                n_jobs=config_data.num_jobs,
                dynamical_system=config_data.dynamical_system,
                dtype=config_data.dtype,
            )
            
                
            giotto_pdg = og.get_persistence_diagrams()
            labels = og._labels
            dgms, labels = pdg_dataset(giotto_pdg, labels, model_name=model_name)

            dgms0 = dgms[0].reshape(-1)
            dgms1 = dgms[1].reshape(-1)
            
            valid_test0 = ((dgms0 < 0).sum() == 0)
            valid_test1 = ((dgms0 > 1).sum() == 0)
            valid_test2 = ((dgms1 < 0).sum() == 0)
            valid_test3 = ((dgms1 > 1).sum() == 0)
            valid_dgms = valid_test0 and valid_test1 and valid_test2 and valid_test3


        train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=0.3)

        if model_name == 'xpert':
            dgms0, dgms1 = dgms

            dgms0_train, dgms1_train, labels_train = dgms0[train_indices], dgms1[train_indices], labels[train_indices]
            dgms0_test, dgms1_test, labels_test = dgms0[test_indices], dgms1[test_indices], labels[test_indices]

            train_dataset = TensorDataset(dgms0_train, dgms1_train, labels_train)
            test_dataset = TensorDataset(dgms0_test, dgms1_test, labels_test)

            train_loader = DataLoader(train_dataset, batch_size=config['bs'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        elif model_name == 'persformer':

            masks = generate_masks(giotto_pdg)
            giotto_pdg = torch.tensor(giotto_pdg, dtype=torch.float32)
            masks = torch.tensor(masks, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            train_dataset = TensorDataset(giotto_pdg[train_indices], masks[train_indices], labels[train_indices])
            test_dataset = TensorDataset(giotto_pdg[test_indices], masks[test_indices], labels[test_indices])

            train_loader = DataLoader(train_dataset, batch_size=config['bs'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

            
        from load_models import load_model_orbit
        from timm.scheduler import CosineLRScheduler

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_orbit(model_name, config).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_config['lr'])
        scheduler = CosineLRScheduler(
                    optimizer,
                    t_initial=hyper_config['epochs'],
                    cycle_mul=1,
                    lr_min=0.05*hyper_config['lr'],
                    cycle_decay=1.,
                    warmup_lr_init=0.05*hyper_config['lr'],
                    warmup_t=hyper_config['warmup_t'],
                    cycle_limit=1,
                    t_in_epochs=True
                )

        best_test_acc = 0
        for epoch in range(hyper_config['epochs']):
            start_time = time.time()
            model.train()
            for data in train_loader:
                optimizer.zero_grad()

                if model_name == 'persformer':
                    dgms, mask, labels = data
                    dgms, mask, labels = dgms.to(device), mask.to(device), labels.to(device)
                    output = model(dgms, mask)

                elif model_name == 'xpert':
                    dgms0, dgms1, labels = data
                    dgms0, dgms1, labels = dgms0.to(device), dgms1.to(device), labels.to(device)
                    output = model(dgms0, dgms1)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

            scheduler.step(epoch)
            

            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                for data in test_loader:
                    if model_name == 'persformer':
                        dgms, mask, labels = data
                        dgms, mask, labels = dgms.to(device), mask.to(device), labels.to(device)
                        output = model(dgms, mask)

                    elif model_name == 'xpert':
                        dgms0, dgms1, labels = data
                        dgms0, dgms1, labels = dgms0.to(device), dgms1.to(device), labels.to(device)
                        output = model(dgms0, dgms1)

                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                test_acc = 100 * correct / total
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch

                end_time = time.time()
                print(f'\rEpoch {epoch} Accuracy: {test_acc:.2f}(best: {best_test_acc:.2f} at {best_epoch}), time: {(end_time - start_time):.3f}', end='')

            if epoch - best_epoch > patience:
                break
        print()
        best_test_accs.append(best_test_acc)
    print(f'mean test accuracy: {np.mean(best_test_accs)} +- {np.std(best_test_accs)}')