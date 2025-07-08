import torch
import torch.nn as nn
import torch.nn.functional as F
from prob_grasp_planner.voxel_ae.voxel_encoder import VoxelEncoder
from prob_grasp_planner.grasp_voxel_planner.grasp_data_loader import convert_to_torch

pkg_path = '/home/mohanraj/robot_WS/src/prob_grasp_planner'
model_dir = '/models/reflex_grasp_inf_models/grasp_voxel_inf_net/'
MODEL_NAME_ = 'grasp_lkh'

VOXEL_ = 'voxel'
OBJ_DIM_ = 'object_dim'
GRASP_CONF_ = 'grasp_config'

class GraspLKHNet(nn.Module):
    def __init__(self):
        super(GraspLKHNet, self).__init__()
        # inspired by Lu et al. IROS 2020 architecture

        # voxel (1,343)
        self.voxel_encoder = VoxelEncoder()
        voxel_ae_model = '/models/reflex_grasp_inf_models/voxel_ae/voxel_encoder'
        self.voxel_encoder.load_state_dict(torch.load(pkg_path+voxel_ae_model))

        # object size + voxel
        self.fc1 = nn.Linear(346, 128)
        self.n1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.n2 = nn.LayerNorm(64)

        # grasp configuration: 1 preshape joint + palm pose (6 numbers)
        self.grasp_fc1 = nn.Linear(7, 14)
        self.gn1 = nn.LayerNorm(14)
        self.grasp_fc2 = nn.Linear(14, 8)
        self.gn2 = nn.LayerNorm(8)

        # object size + voxel + grasp configuration 
        self.fc3 = nn.Linear(72, 64)
        self.n3 = nn.LayerNorm(64)
        self.fc4 = nn.Linear(64, 32)
        self.n4 = nn.LayerNorm(32)
        self.fc5 = nn.Linear(32,2)

        self.dtype = torch.float32
        self.device = torch.device('cuda')


    def forward(self, sample):
        voxel = sample["voxel"]
        object_dim = sample["object_dim"]

        grasp_config = sample["grasp_config"] # [6 + 8]
        # 6 = location + orientation (euler)
        # 16 = 4 fingers 4 joints (order: index, middle, ring, thumb)
        grasp_config.float()

        x = self.voxel_encoder(voxel)

        # voxel + object dimensions
        x = torch.cat((x, object_dim), dim=1)
        x = self.fc1(x)
        x = self.n1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = self.n2(x)
        x = F.elu(x)

        # grasp configuration
        g = self.grasp_fc1(grasp_config)
        g = self.gn1(g)
        g = F.elu(g)
        g = self.grasp_fc2(g)
        g = self.gn2(g)
        g = F.elu(g)
   
        # grasp config + voxel + object dimensions
        x = torch.cat((x, g), dim=1)
        x = self.fc3(x)
        x = self.n3(x)
        x = F.elu(x)
        x = self.fc4(x)
        x = self.n4(x)
        x = F.elu(x)

        x = self.fc5(x) # 32 -> 2

        #print(F.softmax(x, dim=1))        
        x = F.log_softmax(x, dim=1) # better numerical properties than vanilla softmax

        return x

    def object_condition(self, voxel, object_dim):
        self.sample = {}
        self.sample[VOXEL_] = voxel
        self.sample[OBJ_DIM_] = object_dim
        self.sample = convert_to_torch(self.sample)

    def query_grasp_lkh(self, grasp_config):
        self.sample[VOXEL_] = self.sample[VOXEL_].detach()
        self.sample[OBJ_DIM_] = self.sample[OBJ_DIM_].detach()
        #grasp_config = torch.from_numpy(grasp_config).type(self.dtype).to(self.device)
        #grasp_config.requires_grad = True
        self.sample[GRASP_CONF_] = grasp_config
        return self(self.sample)[0][0]

    def query_grasp_lkh_grad(self, grasp_config):
        self.sample[VOXEL_] = self.sample[VOXEL_].detach()
        self.sample[OBJ_DIM_] = self.sample[OBJ_DIM_].detach()
        grasp_config = grasp_config.detach()
        grasp_config.requires_grad = True
        lkh = self.query_grasp_lkh(grasp_config)
        return torch.autograd.grad(lkh, grasp_config)[0].detach().cpu().numpy()

    def query_num_grad(self, grasp_config, delt = 1e-6):
        grasp_config = grasp_config.detach()
        lkh = self.query_lkh(grasp_config)[0][1]
        grad = torch.zeros(grasp_config.shape, device=self.device)
        for i in range(7):
            grasp_config_f = torch.clone(grasp_config).detach()
            grasp_config_f[0][i] += delt
            lkh_f = self.query_lkh(grasp_config_f)[0][1]
            grad[0][i] =  lkh_f - lkh
            grad[0][i] /= 2*delt
        return grad.detach().cpu().numpy()

def build_lkh():
    net = GraspLKHNet()
    net.load_state_dict(torch.load(pkg_path+model_dir+MODEL_NAME_))
    net = net.to(net.device)
    net.eval()
    return net

def main():
    #net = GraspLKHNet()
    #for param_tensor in net.state_dict():
    #    print(param_tensor)

    net = build_lkh()
    import numpy as np
    import time

    voxel = np.random.randint(0,2,(1,1,32,32,32))
    voxel *= 0
    voxel[0,0,1,:,:] = 1
    voxel[0,0,:,2,:] = 1
    voxel[0,0,:,:,30] = 1
    odims = np.random.rand(1,3)
    odims[0,0] = 0.2
    odims[0,1] = 0.2
    odims[0,2] = 0.4
    grasp_conf = np.random.rand(1,7)
    grasp_conf[0,3] = 0.0
    grasp_conf[0,4] = 1.57
    grasp_conf[0,5] = 0.0
    grasp_conf[0,0] = 0.0
    grasp_conf[0,1] = 0.0
    grasp_conf[0,2] = odims[0,2] * 2
    grasp_conf = torch.from_numpy(grasp_conf).type(net.dtype).to(net.device)
    net.object_condition(voxel, odims)
    st = time.time()
    for i in range(1000):
        #grasp_conf = np.random.rand(1,7)
        #grasp_conf = torch.from_numpy(grasp_conf).type(net.dtype).to(net.device)
        #net.query_grasp_lkh(grasp_conf)
        grasp_conf[0,2] -= 0.01
        #print(grasp_conf[0,2])
        net.query_grasp_lkh(grasp_conf)
    exit(0)
    pt1 = time.time()
    for i in range(1000):
        grasp_conf = np.random.rand(1,7)
        grasp_conf = torch.from_numpy(grasp_conf).type(net.dtype).to(net.device)
        grad = net.query_lkh_grad(grasp_conf)
        grad_fd = net.query_num_grad(grasp_conf)
        print('similarity: ', grad @ grad_fd.T)
    pt2 = time.time()
    print('Query Time: ', pt1-st, (pt1-st)/1000)
    print('Grads Time: ', pt2-pt1, (pt2-pt1)/1000)

if __name__ == "__main__":
    main()
