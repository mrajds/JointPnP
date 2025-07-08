import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from voxel_ae_dnn import VoxelAE
from dataset_loader import VoxelAEDataset


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
    dataset_path = "/mnt/Voxel_CF_numpy/"

    model = VoxelAE().to(device)
    model.load_state_dict(torch.load("/home/mmatak/voxel_ae-final"))

    test_dataset = VoxelAEDataset(dataset_path + "Test/")
    test_loader = torch.utils.data.DataLoader(test_dataset)
   
    test_loss = test(model, device, test_loader)
   
if __name__ == "__main__":
    main()
