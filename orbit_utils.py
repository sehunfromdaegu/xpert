
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader

def quantization_np(rdgms, grid_size=50):
    '''
    Read rotated persistence diagrams, and outputs a matrix of size (grid_size,grid_size).
    Rotated diagrams are quantized into grids, where the maximum values of x and y are given by 'x_ranges' and 'y_ranges'.
    Each entry of the matrix contains the number of points in the diagram that correspond to each grid.
    Outputs the matrix.
    
    rdgms : (bs, n_pts, 2)
    '''
    bs, n_pts, _ = rdgms.shape
    quantized_matrices = np.zeros((bs, grid_size, grid_size), dtype=int)
    
    for i in range(bs):
        births = rdgms[i, :, 0]
        pers = rdgms[i, :, 1]
        
        # Filter out points with persistence 0
        valid_mask = (pers != 0)
        births = births[valid_mask]
        pers = pers[valid_mask]
        
        if births.size == 0:
            continue

        max_birth = births.max()
        max_per = pers.max()

        x_range = (0, 1.1 * max_birth)
        y_range = (0, 1.1 * max_per)

        if max_birth == 0:
            x_range = (0, 1)


        x_step = (x_range[1] - x_range[0]) / grid_size
        y_step = (y_range[1] - y_range[0]) / grid_size


        x_indices = np.floor((births - x_range[0]) / x_step).astype(int)
        y_indices = np.floor((pers - y_range[0]) / y_step).astype(int)

        x_indices = np.clip(x_indices, 0, grid_size - 1)
        y_indices = np.clip(y_indices, 0, grid_size - 1)

        # Count the points in each grid cell
        for x_idx, y_idx in zip(x_indices, y_indices):
            quantized_matrices[i, y_idx, x_idx] += 1
    
    return quantized_matrices


def quantization(rdgms, grid_size=50):
    '''
    Read rotated persistence diagrams, and outputs a matrix of size (grid_size, grid_size).
    Rotated diagrams are quantized into grids, where the maximum values of x and y are given by 'x_ranges' and 'y_ranges'.
    Each entry of the matrix contains the number of points in the diagram that correspond to each grid.
    Outputs the matrix.
    
    rdgms : (bs, n_pts, 2)
    output : (bs, grid_size, grid_size)
    '''
    bs, n_pts, _ = rdgms.shape
    quantized_matrices = torch.zeros((bs, grid_size, grid_size), dtype=torch.float32)
    
    births = rdgms[:, :, 0]
    pers = rdgms[:, :, 1]
    
    # Filter out points with persistence 0
    valid_mask = (pers != 0)
    births = births * valid_mask
    pers = pers * valid_mask
    
    max_birth = births.max(dim=1, keepdim=True)[0]
    max_per = pers.max(dim=1, keepdim=True)[0]

    x_range = torch.where(max_birth == 0, torch.ones_like(max_birth), 1.1 * max_birth)
    y_range = 1.1 * max_per

    x_step = x_range / grid_size
    y_step = y_range / grid_size

    x_indices = ((births / x_step).floor())
    y_indices = ((pers / y_step).floor())

    x_indices = torch.clamp(x_indices, 0, grid_size - 1)
    y_indices = torch.clamp(y_indices, 0, grid_size - 1)

    for i in range(bs):
        idx_comb = (y_indices[i] * grid_size + x_indices[i]).to(torch.int64)
        idx_comb = idx_comb[valid_mask[i]]
        counts = torch.bincount(idx_comb, minlength=grid_size * grid_size).float()
        quantized_matrices[i] = counts.view(grid_size, grid_size)
    
    return quantized_matrices



def visualize_quantization(quantized_matrix):
    plt.imshow(quantized_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Quantized Persistence Diagram Heatmap')
    plt.xlabel('Birth')
    plt.ylabel('Persistence')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.show()


def rotate_diagrams(dgms):
    rotated_dgms = dgms.copy()
    rotated_dgms[:,:,1] = rotated_dgms[:,:,1] - rotated_dgms[:,:,0]
    return rotated_dgms

def pdg_reshape(giotto_pdg):
    '''
    reshape the pdg (bs, n, 4) to (bs, n, 2)
    n = max number of points in the diagrams
    '''
    pds = copy.deepcopy(giotto_pdg)
    pds0 = pds[:,:999][:,:,:2]
    pds1 = pds[:,999:][:,:,:2]
    return pds0, pds1


def remove_zero_pers(pds1):
    """
    Remove the points with persistence 0
    """
    # Use list comprehension and np.where within the loop to filter directly
    return [pd[np.where(pd[:,0] != pd[:,1])] for pd in pds1]

def generate_masks(giotto_pdg):
    '''
    giotto_pdg : (bs, n , 4)
    giotto_pdg is padded with trivial points whose birth and death are the same.
    This function generates masks for such points.
    '''
    masks = (giotto_pdg[:,:,0] == giotto_pdg[:,:,1])
    return masks


def pdg_dataset(giotto_pdg, labels, model_name='xpert'):
    assert model_name in ['xpert', 'persformer', 'atol']

    pds0, pds1 = pdg_reshape(giotto_pdg)

    if model_name == 'persformer':
        return giotto_pdg, labels
    
    if model_name == 'xpert':
        rdgms0 = rotate_diagrams(pds0)
        rdgms0 = torch.tensor(rdgms0)
        rdgms1 = rotate_diagrams(pds1)
        rdgms1 = torch.tensor(rdgms1)
        labels = torch.tensor(labels)

        return (rdgms0, rdgms1), labels
    
    if model_name == 'atol':
        return (pds0, remove_zero_pers(pds1)), labels