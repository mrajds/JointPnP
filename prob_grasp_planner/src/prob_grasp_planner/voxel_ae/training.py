import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from voxel_ae_dnn import VoxelAE
from dataset_loader import VoxelAEDataset


MIN_VALUE = 1e-7
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    avg_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        for key in sample:
            sample[key] = sample[key].to(device) optimizer.zero_grad()
        output = model(sample["partial"])
        output = torch.clip(output, MIN_VALUE, 1.0-MIN_VALUE)
        target = sample["full"]
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(sample), len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))
        avg_loss = avg_loss + loss.item()
    avg_loss /= len(train_loader)
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for sample in test_loader:
            for key in sample:
                sample[key] = sample[key].to(device)
            target = sample["full"]
            output = model(sample["partial"])

            test_loss += F.binary_cross_entropy(output, target)
    test_loss /= len(test_loader.dataset)
    test_loss = test_loss.item()
    print("test loss: ", test_loss)
    return test_loss

def main():
    torch.manual_seed(2021)
    device = torch.device("cuda")
    path = "/mnt/Voxel_CF_numpy/"

    train_dataset = VoxelAEDataset(path + "Train/") 
    test_dataset = VoxelAEDataset(path + "Validation/")
    model = VoxelAE().to(device)

    test_loader = torch.utils.data.DataLoader(test_dataset)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_epoch = -1
    min_test_loss = None
    nr_epochs = 100
    for epoch in range(1, nr_epochs):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        if min_test_loss is None or test_loss < min_test_loss:
            best_epoch = epoch
            min_test_loss = test_loss
            torch.save(model.state_dict(), "/home/mmatak/voxel_ae-" + str(epoch)) 

        if epoch == 20 or epoch == 80:
            for g in optimizer.param_groups:
                g['lr'] /= 10

    print("best test loss: " + str(min_test_loss))
    print("best epoch: " + str(best_epoch))

    torch.save(model.state_dict(), "/home/mmatak/voxel_ae-final") 


if __name__ == "__main__":
    main()
