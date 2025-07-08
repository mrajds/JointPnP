import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
from prob_grasp_planner.grasp_voxel_planner.grasp_data_loader import GraspDataLoader, \
    convert_to_torch
from prob_grasp_planner.voxel_ae.voxel_encoder import VoxelEncoder
from matplotlib import pyplot

pkg_path = '/home/mohanraj/robot_WS/src/prob_grasp_planner'
prior_model_dir = '/models/reflex_grasp_inf_models/grasp_voxel_inf_prior/'

N_COMPS_ = 2

class MDN(nn.Module):
    def __init__(self, n_gaussians=3, output_dim=14, gpu=True):
        super(MDN, self).__init__()
        self.n_gaussians = n_gaussians
        self.n_outputs = output_dim
        if gpu:
            self.dtype = torch.float32
            self.device = torch.device("cuda")
        else:
            self.dtype = torch.float32
            self.device = torch.device("cpu")

        # inspired by Lu et al. IROS 2020 architecture
        # voxel
        self.voxel_encoder = VoxelEncoder()
        voxel_ae_model = '/models/reflex_grasp_inf_models/voxel_ae/voxel_encoder'
        self.voxel_encoder.load_state_dict(torch.load(pkg_path+voxel_ae_model))

        # object size + voxel
        self.fc1 = nn.Linear(346, 128)
        self.n1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 32)
        self.n2 = nn.LayerNorm(32)

        last_layer_size = 32 # MDN parameters
        self.pi = nn.Linear(last_layer_size, self.n_gaussians)
        self.mu = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs) 

        # lower triangle of covariance matrix (below the diagonal)
        self.L = nn.Linear(last_layer_size, int(0.5 * self.n_gaussians * self.n_outputs * (self.n_outputs - 1)))
        # the diagonal of covariance matrix
        self.L_diagonal = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs)

        self.obj_mdn = None
        self.obj_dim = None


    def forward(self, sample):
        voxel = sample["voxel"]
        object_dim = sample["object_dim"]
        x = voxel
        x = self.voxel_encoder(voxel)

        # voxel + object dimensions
        x = torch.cat((x, object_dim), dim=1)
        x = self.fc1(x)
        x = self.n1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.n2(x)
        x = F.relu(x) # (n, 32)

        #pi = nn.functional.elu(self.pi(x))
        pi = nn.functional.softmax(self.pi(x), -1)
        mu = self.mu(x).reshape(-1, self.n_outputs, self.n_gaussians)
        #L_diagonal = torch.exp(self.L_diagonal(x)).reshape(-1, self.n_outputs, self.n_gaussians)
        L_diagonal = nn.functional.elu(self.L_diagonal(x)).reshape(-1, self.n_outputs, self.n_gaussians) + 1

        # below the main diagonal
        L = self.L(x).reshape(-1, int(0.5 * self.n_outputs * (self.n_outputs - 1)), self.n_gaussians)

        return pi, mu, L, L_diagonal

    def loss_fn(self, pi, mu, L, L_d, target, save_input_gradient = False):
        if save_input_gradient:
            target.requires_grad = True
        result = torch.zeros(target.shape[0], self.n_gaussians).to(self.device)
        tril_idx = np.tril_indices(self.n_outputs, -1) # -1 because it's below the main diagonal
        diag_idx = np.diag_indices(self.n_outputs)

        for idx in range(self.n_gaussians):
            tmp_mat = torch.zeros(target.shape[0], self.n_outputs, self.n_outputs).to(self.device)
            tmp_mat[:, tril_idx[0], tril_idx[1]] = L[:, :, idx]
            tmp_mat[:, diag_idx[0], diag_idx[1]] = L_d[:, :, idx]
            try:
                mvgaussian = MultivariateNormal(loc=mu[:, :, idx], scale_tril=tmp_mat)
            except ValueError as ve:
                print("error: ")
                print(ve)
                print("loc:")
                print(mu[:,:,idx])
                print("mu:")
                print(mu)
            result_per_gaussian = mvgaussian.log_prob(target)
            result[:, idx] = result_per_gaussian + pi[:, idx].log()
        result = -torch.mean(torch.logsumexp(result, dim=1))

        # when optimizing over q using non-torch optimizer (check planner.py)
        if save_input_gradient:
            result.backward(retain_graph=True)
            self.q_grad = target.grad.cpu().numpy().astype('float64')

        return result

    def train_network(self, data_loader, optimizer, nepoch=90, batch_size=1):
        self.train()
        lossHistory = []

        #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.93)
        #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[60, 80, 100, 150, 200], 0.1) # Full data schedule
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[150, 200], 0.1) # Success only schedule

        # freeze weights for encoder
        for name, param in self.named_parameters():
            if param.requires_grad and "voxel" in name:
                param.requires_grad = False

        progress = tqdm(range(nepoch))

        for epoch in progress:
            while data_loader.epochs_completed <= epoch:
                sample = self.convert_to_torch(data_loader.next_batch(batch_size))
                optimizer.zero_grad()
                #pi, mu, L, L_d = self(sample)
                #grasp2 = sample["grasp_config"].detach()
                ##loss = self.loss_fn(pi, mu, L, L_d, sample["grasp_config"], True)
                #sample["grasp_config"].zero_grad()
                self.object_condition_mdn(sample, batch_size)
                loss = self.query_grasp_prior(sample["grasp_config"])
                #grad = self.query_grasp_prior_grad(sample["grasp_config"])
                #print(loss2 - loss)
                loss.backward()
                optimizer.step()
                lossHistory.append(loss.item())
                progress.set_description(f"Epoch {epoch} Loss: {loss.item()}")

            #if epoch != 0 and epoch % 10 == 0:
                #print("epoch: ", epoch, "loss: ", loss.item())
                #torch.save(self.state_dict(), "" + str(epoch))
            #print(self.pi.weight)
            #print(self.mu.weight)
            lr_scheduler.step()
        #exit(0)

        print("Training finished, final loss: {}".format(loss.item()))
        return lossHistory

    def sample(self, sample, nr_samples=1):
        '''
        Return one sample from the most impactful distribution.
        :param sample: single input, dictionary with numpy values
        '''
        sample = self.convert_to_torch(sample)
        pi, mean, L, L_diagonal = self(sample)
        return self.sample_from_mdn(pi, mean, L, L_diagonal, nr_samples), (pi, mean, L, L_diagonal)

    def convert_to_torch(self, sample):
        return convert_to_torch(sample, self.dtype, self.device)
        
    def sample(self):
        mv_gaussians, pi, comp_prior_template = self.obj_mdn
        mix = Categorical(pi)
        idx = mix.sample().item()
        result = mv_gaussians[idx].sample()
        while self.check_sample(result):
            result = mv_gaussians[idx].sample()
        return result

    def sample_explicit(self, idx):
        mv_gaussians, pi, comp_prior_template = self.obj_mdn
        #mix = Categorical(pi)
        #idx = mix.sample().item()
        result = mv_gaussians[idx].sample()
        while self.check_sample(result):
            result = mv_gaussians[idx].sample()
        return result

    def check_sample(self, sample):
        return torch.all(sample[:, :3] > -self.obj_dim[:, :]/2) and torch.all(sample[:, 3] < self.obj_dim[:,:]/2)

    def object_condition_mdn(self, grasp_object, batch_size=1):
        grasp_object = self.convert_to_torch(grasp_object)
        self.obj_dim = grasp_object["object_dim"]
        pi, mu, L, L_d = self(grasp_object)

        if batch_size==1:
            print(f"MDN Weights: {pi}")
            print(f"Means: {mu}")
            print(f"Vars: {L_d}")
        
        tril_idx = np.tril_indices(self.n_outputs, -1) # -1 because it's below the main diagonal
        diag_idx = np.diag_indices(self.n_outputs)

        mv_gaussians = []

        for idx in range(self.n_gaussians):
            tmp_mat = torch.zeros(batch_size, self.n_outputs, self.n_outputs).to(self.device)
            tmp_mat[:, tril_idx[0], tril_idx[1]] = L[:, :, idx]
            tmp_mat[:, diag_idx[0], diag_idx[1]] = L_d[:, :, idx]
            try:
                mv_gaussians.append(MultivariateNormal(loc=mu[:, :, idx], scale_tril=tmp_mat))
            except ValueError as ve:
                print("error: ")
                print(ve)
                print("loc:")
                print(mu[:,:,idx])
                print("mu:")
                print(mu)
        self.obj_mdn = (mv_gaussians, pi, torch.zeros(batch_size, self.n_gaussians, device=self.device))
        return mu.detach().cpu().numpy(), pi.detach().cpu().numpy()

    def query_grasp_prior(self, grasp_config, comps=False):
        mv_gaussians, pi, comp_prior_template = self.obj_mdn
        comp_prior = torch.clone(comp_prior_template).detach()
        #pi = pi.detach()
        #comp_prior = comp_prior.detach()
        
        for i in range(self.n_gaussians):
            comp_prior[:, i] = mv_gaussians[i].log_prob(grasp_config) + pi[:, i].log()
            #comp_prior[:, i] = comp_prior[:, i].detach()
        prior = torch.mean(-torch.logsumexp(comp_prior, dim=1))
        if comps:
            prior = -torch.logsumexp(comp_prior, dim=1).detach().cpu().numpy()
            return prior, (comp_prior.detach().cpu().numpy(), pi.detach().cpu().numpy())
        return prior

    def query_grasp_prior_grad(self, grasp_config):
        grasp_config = grasp_config.detach()
        grasp_config.requires_grad = True
        prior = self.query_grasp_prior(grasp_config)
        return torch.autograd.grad(prior, grasp_config)[0].detach().cpu().numpy().astype('float64')

    def query_num_prior_grad(self, grasp_config, delt = 1e-6):
        grasp_config = grasp_config.detach()
        prior = self.query_grasp_prior(grasp_config)
        grad = torch.zeros(grasp_config.shape, device=self.device)
        for i in range(self.n_outputs):
            grasp_config_f = torch.clone(grasp_config).detach()
            grasp_config_f[i] += delt
            prior_f = self.query_grasp_prior(grasp_config_f)
            grad[i] =  prior_f - prior
            grad[i] /= 2*delt
        return grad.detach().cpu().numpy()
        
    
def build_MDN():
    net = MDN(n_gaussians=N_COMPS_, output_dim=7)
    net.load_state_dict(torch.load(pkg_path+prior_model_dir+'mdn'))
    net = net.to(net.device)
    net.eval()
    return net

def train():
    net = MDN(n_gaussians=N_COMPS_, output_dim=7)
    net = net.to(net.device)
    print("device used for training: ", net.device)
    #net.apply(init_weights)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #path = "/mnt/data_collection/"

    num_epoch = 250
    #train_dataset = TactileGraspDataset(path, positive_only=True, preshape_data=True) 
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    train_data_path = '/home/mohanraj/reflex_grasp_data/processed/batch3/' + \
                        'grasp_voxelized_data.h5'
    grasp_loader = GraspDataLoader(train_data_path)
    #print('nr samples: ', len(train_dataset))
    losses = net.train_network(data_loader = grasp_loader, optimizer = optimizer, nepoch=num_epoch, batch_size=1024)

    pyplot.plot(range(len(losses)), losses)
    pyplot.show()

    torch.save(net.state_dict(), pkg_path+prior_model_dir+'mdn')

def main():
    train()
 
    #path = "/mnt/data_collection/"
    #train_dataset = TactileGraspDataset(path, positive_only=True, preshape_data=True) 
    ##data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)

    train_data_path = '/home/mohanraj/reflex_grasp_data/processed/batch3/' + \
                        'success_grasp_voxelized.h5'
    grasp_loader = GraspDataLoader(train_data_path)
    
    # all elements must be 0 or 1
    voxel = torch.zeros((1,1,32,32,32))
    obj_dim = torch.rand((1,3))
    sample = {}
    sample["voxel"] = voxel
    sample["object_dim"] = obj_dim
   
    net = MDN(n_gaussians=N_COMPS_, output_dim=7)
    net.load_state_dict(torch.load(pkg_path+prior_model_dir+'mdn'))
    net = net.to(net.device)
    net.eval()

    net.object_condition_mdn(sample)

    config = net.sample()
    prior = net.query_grasp_prior(config)
    print(f"Sample generated with prior: {prior}")

def query_test(f, N=100):
    grasp = torch.rand(7, device = torch.device('cuda'))
    st = time.time()
    for i in range(N):
        grasp = torch.rand(7, device=torch.device('cuda'))
        f(grasp)
        end = time.time() - st
    return N, end/N

def test_grad(net, means, N=10):
    grasp = torch.rand(7, device = torch.device('cuda'))
    NMeans = means.shape[2]
    for i in range(N):
        if i < NMeans:
            grasp = torch.from_numpy(means[0,:,i]).type(torch.float32).to(torch.device('cuda'))
        else:
            idx = np.random.randint(7)
            grasp[idx] += torch.rand(1, device = torch.device('cuda'))[0]
        grad = net.query_grasp_prior_grad(grasp)
        fd = net.query_grasp_prior_grad(grasp, backmode=True)
        print(np.vstack((grad, fd, grad-fd)).T)
        #grasp = torch.rand(7, device = torch.device('cuda'))


if __name__ == "__main__":
    main()
    #exit(0)
    import time
    net = build_MDN()
    voxel = torch.randint(0,2,(1,1,32,32,32))
    obj_dim = torch.rand((1,3))
    sample = {}
    sample["voxel"] = voxel
    sample["object_dim"] = obj_dim

    means, weights = net.object_condition_mdn(sample)
    print(net.obj_mdn[1])
    print(query_test(net.query_grasp_prior, 1000))
    print(query_test(net.query_grasp_prior_grad, 1000))
    print(query_test(net.query_grasp_prior, 1000))
    #test_grad(net, means, 5)
