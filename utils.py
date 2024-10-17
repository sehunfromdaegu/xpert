import torch
import copy
import torch.nn as nn

def convert_to_epd_list(pd_ds):
    epds = []
    labels = []
    for i in range(pd_ds.len):
        epd = pd_ds[i][0]._data
        label = pd_ds[i][1]
        epd_categorical = epd[:,2:].argmax(dim=1)
        epd_list = [epd[:,:2][epd_categorical==i] for i in range(4)]

        epds.append(epd_list)
        labels.append(label)

    return epds, labels

def separate_data_labels(pd_ds):
    return [pd_ds[i][0] for i in range(pd_ds.len)], [pd_ds[i][1] for i in range(pd_ds.len)]

def rotate_epds(epds):
    rotated_epds = []
    for epd_list in epds:
        rotated_epd_list = []
        for dgm in epd_list:
            rdgm = copy.deepcopy(dgm)
            rdgm[:,1] -= rdgm[:,0]
            rotated_epd_list.append(rdgm)

        rotated_epds.append(rotated_epd_list)

    return rotated_epds

def pixelization(repd_list, grid_size=50, device='cpu'):
    quantized_matrices = torch.zeros((4, grid_size, grid_size), dtype=torch.float32, device=device)
    
    # find max birth and max pers
    max_birth = 0.
    min_birth = 100.
    max_pers = 0.
    
    for i in range(4):
        rdgm = repd_list[i]
        births = rdgm[:,0]
        pers = rdgm[:,1]

        # Check if births is empty
        if births.numel() == 0:
            max_birth = max(max_birth, 0)
            min_birth = min(min_birth, 0)
        else:
            max_birth = max(max_birth, births.max())
            min_birth = min(min_birth, births.min())
        
        # Check if pers is empty
        if pers.numel() == 0:
            max_pers = max(max_pers, 0)
        else:
            max_pers = max(max_pers, pers.max())

    for i in range(4):
        rdgm = repd_list[i]
        if rdgm.shape[0] == 0:
            continue
        births = rdgm[:,0]
        pers = rdgm[:,1]
        
        x_range = torch.where(max_birth == 0, torch.ones_like(max_birth), 1.1 * max_birth)
        y_range = 1.1 * max_pers

        epsilon = 1e-8
        x_step = (x_range + epsilon) / grid_size
        y_step = (y_range + epsilon) / grid_size
            
        x_indices = ((births)/ x_step).floor()
        y_indices = (pers / y_step).floor()

        x_indices = torch.clamp(x_indices, 0, grid_size - 1)
        y_indices = torch.clamp(y_indices, 0, grid_size - 1)

        idx_comb = (y_indices * grid_size + x_indices).to(torch.int64)
        counts = torch.bincount(idx_comb, minlength=grid_size * grid_size).float()
        quantized_matrices[i] = counts.view(grid_size, grid_size)

    return quantized_matrices


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler = None):
    """
    Trains the model for one epoch, optionally adjusting learning rate with a scheduler.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): The device (CPU/GPU) to run the training on.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler to adjust the LR. Defaults to None.

    Returns:
        Tuple[float, float]: Tuple containing average training loss and training accuracy for this epoch.
    """
    
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for data in train_loader:
        # Move data to the device
        data = data.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, data.y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = outputs.max(1)  # Get predictions
        correct_predictions += predicted.eq(data.y).sum().item()  # Count correct predictions
        total_samples += data.y.size(0)  # Count total samples

    # Calculate average loss and accuracy for the epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # Step the scheduler after each epoch if provided
    if scheduler is not None:
        scheduler.step(epoch)

    return avg_loss, accuracy


@torch.no_grad()
def test(model, test_loader, criterion, device):
    """
    Evaluates the model on the test dataset for one epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device (CPU/GPU) to run the evaluation on.

    Returns:
        Tuple[float, float]: Tuple containing average test loss and test accuracy for this epoch.
    """
    
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for data in test_loader:
        # Move data to the device
        data = data.to(device)

        # Forward pass (no gradient computation)
        outputs = model(data)
        loss = criterion(outputs, data.y).item()

        # Accumulate loss
        running_loss += loss

        # Calculate accuracy
        _, predicted = outputs.max(1)  # Get the index of the max log-probability
        correct_predictions += predicted.eq(data.y).sum().item()  # Count correct predictions
        total_samples += data.y.size(0)  # Count total samples

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(test_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    return avg_loss, accuracy









# @torch.no_grad()
# def test(model, loader, criterion):
#     device = next(model.parameters()).device
#     model.eval()
#     total_loss = 0.
#     correct = 0
#     total = 0
    
#     for data in loader:
#         ppd, target = data
#         ppd = ppd.to(device)  # Move data to the GPU
#         target = target.to(device)

#         out = model(ppd)
#         loss = criterion(out, target).item()
#         total_loss += loss / len(loader)
        
#         # Assuming the output `out` and labels `data.y` are one-hot encoded or logits
#         pred = out.argmax(dim=1)  # Get the index of the max log-probability
#         correct += pred.eq(target).sum().item()  # Count correct predictions
#         total += target.size(0)  # Total number of samples

#     accuracy = correct / total if total > 0 else 0
#     return total_loss, accuracy