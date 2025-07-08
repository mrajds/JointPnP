import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VoxelEncoder(nn.Module):
    def __init__(self, latent_dim = 100):
        super(VoxelEncoder, self).__init__()
        self.latent_dim = latent_dim
        # inspired by Lu et al. IROS 2020 architecture

        # encoder 
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, stride=2)
        self.voxel_fc = nn.Linear(7*7*7*64, 343)
        
        # latent space 
        #self.to_latent = nn.Linear(343, self.latent_dim)

    def forward(self, voxel):
        x = voxel

        # encoder
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.voxel_fc(x)
        x = F.elu(x)

        # latent
        #x = self.to_latent(x)
        #x = F.elu(x)
        
        return x

    def evaluate(self, voxel_np, return_np=True):
        # _np means it has to be numpy object
        voxel = torch.from_numpy(voxel_np.astype('float32')) # shape: (32,32,32)
        voxel = voxel.unsqueeze(0) # 1 x 32 x 32 x 32
        voxel = voxel.unsqueeze(0) # 1 x 1 x 32 x 32 x 32

        output = self(voxel)

        if return_np:
            return output.detach().numpy()[0] # same as above
        else:
            return output
 

def main():
    '''
    Loads stored autoencoder and extracts relevant weights to encoder
    '''
    from voxel_ae_dnn import VoxelAE
    from dataset_loader import VoxelAEDataset

    torch.manual_seed(2021)
    device = torch.device("cuda")
    
    voxel_ae = VoxelAE()
    voxel_ae.load_state_dict(torch.load("/home/mmatak/voxel_ae-final"))
    voxel_encoder = VoxelEncoder()

    voxel_ae_state_dict = voxel_ae.state_dict()
    for param_tensor in voxel_encoder.state_dict():
        voxel_encoder.state_dict()[param_tensor].copy_(voxel_ae_state_dict[param_tensor])
    
    torch.save(voxel_encoder.state_dict(), "/home/mmatak/voxel_encoder")

    voxel_encoder.load_state_dict(torch.load("/home/mmatak/voxel_encoder"))
    for param_tensor in voxel_encoder.state_dict():
        assert torch.equal(voxel_encoder.state_dict()[param_tensor], voxel_ae.state_dict()[param_tensor])

    voxel = np.ones((32,32,32))
     
    voxel = torch.from_numpy(voxel.astype('float32')) # shape: (32,32,32)
    voxel = voxel.unsqueeze(0) # 1 x 32 x 32 x 32
    voxel = voxel.unsqueeze(0) # 1 x 1 x 32 x 32 x 32
    output = voxel_encoder(voxel)
    assert output.shape == (1,343), output.shape
 
if __name__ == "__main__":
    main()
