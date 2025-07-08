import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelAE(nn.Module):
    def __init__(self, latent_dim = 100):
        super(VoxelAE, self).__init__()
        self.latent_dim = latent_dim
        # inspired by Lu et al. IROS 2020 architecture

        # encoder 
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, stride=2)
        self.voxel_fc = nn.Linear(7*7*7*64, 343)
        
        # latent space 
        self.to_latent = nn.Linear(343, self.latent_dim)

        # decoder
        self.from_latent = nn.Linear(self.latent_dim, 343)
        self.deconv1 = nn.Conv3d(1, 64, 3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(64, 32, 3, stride=2)
        self.deconv3 = nn.Conv3d(32, 16, 3, padding=1)
        self.deconv4 = nn.ConvTranspose3d(16, 8, 4, stride=2)
        self.deconv5 = nn.Conv3d(8, 1, 3, padding=1)

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
        x = self.to_latent(x)
        x = F.elu(x)
        
        # decoder
        x = self.from_latent(x)
        x = F.elu(x)
        x = x.unflatten(1, (1,7,7,7))
        x = self.deconv1(x)
        x = F.elu(x)
        x = self.deconv2(x)
        x = F.elu(x)
        x = self.deconv3(x)
        x = F.elu(x)
        x = self.deconv4(x)
        x = F.elu(x)
        x = self.deconv5(x)
        x = torch.sigmoid(x)
   
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
 
MIN_VALUE = 1e-7
def train(model, sample, target, optimizer):
    model.train()
    avg_loss = 0
    for i in range(100):
        optimizer.zero_grad()
        output = model(sample)
        output = torch.clip(output, MIN_VALUE, 1.0-MIN_VALUE)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(loss.item())

   
def main():
    import numpy as np
    voxel = np.ones((32,32,32))
     
    voxel = torch.from_numpy(voxel.astype('float32')) # shape: (32,32,32)
    voxel = voxel.unsqueeze(0) # 1 x 32 x 32 x 32
    voxel = voxel.unsqueeze(0) # 1 x 1 x 32 x 32 x 32
    target = voxel
    
    model = VoxelAE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=1.0)

    train(model, voxel, target, optimizer)
if __name__ == "__main__":
    main()


