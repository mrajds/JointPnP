#!/usr/bin/env python
'''
Grasp service calls to make only voxel inference;
get priors for initializations, get grasp gradients for the optimization.
'''
import rospy
import rospkg
import numpy as np
from point_cloud_segmentation.msg import GraspObject
from point_cloud_segmentation.srv import SegmentGraspObjectResponse
#from grasp_pipeline.srv import *
from prob_grasp_planner.srv import GraspVoxelInfer, GraspVoxelInferRequest, \
    GraspObjective, GraspObjectiveRequest, GraspObjectiveGrad, GraspObjectiveGradRequest, \
    GraspPriorGen, GraspPriorGenRequest
from sensor_msgs.msg import JointState
from prob_grasp_planner.grasp_voxel_planner.grasp_mdn_prior import build_MDN
from prob_grasp_planner.grasp_voxel_planner.grasp_success_network import build_lkh
import time
import torch
import warnings
import inspect
from gpu_sd_map.ros_transforms_lib import convert_array_to_pose
from ll4ma_pick_n_place.visualizations import broadcast_tf

#Importing grasp_libraries
#TODO: Better way is to use setup.py via catkin.
#rospack = rospkg.RosPack()
#rospack.list()
#prob_grasp_path = rospack.get_path('prob_grasp_planner')
#sys.path.append(prob_grasp_path + '/src/grasp_voxel_planner')
#from prob_grasp_planner.grasp_voxel_planner.grasp_voxel_inference import GraspVoxelInference

def warn_py2tf_call():
    warnings.warn(f"Py2 API Called {inspect.stack()}", DepricationWarning)

class GraspUtils:
    '''
    Grasp pipeline and prob_grasp_planner simplified interface.
    Place to add other grasp planners.
    '''
    def __init__(self, init_node = False):
        if init_node:
            rospy.init_node('pnp_grasp_utils_node')
        QUERY_GRASP_PRIOR_SERVICE_ = "query_grasp_prior"
        QUERY_GRASP_COST_SERVICE_ = "query_grasp_posterior"
        QUERY_GRASP_GRAD_SERVICE_ = "query_grasp_posterior_grad"
        #rospy.wait_for_service('grasp_voxel_infer')
        #rospy.wait_for_service(QUERY_GRASP_PRIOR_SERVICE_)
        #rospy.wait_for_service(QUERY_GRASP_COST_SERVICE_)
        #rospy.wait_for_service(QUERY_GRASP_GRAD_SERVICE_)
        #self.Service = rospy.ServiceProxy('grasp_voxel_infer', GraspVoxelInfer)
        #self.query_grasp_cost_client = rospy.ServiceProxy(QUERY_GRASP_COST_SERVICE_, GraspObjective)
        #self.query_grasp_grad_client = rospy.ServiceProxy(QUERY_GRASP_GRAD_SERVICE_, GraspObjectiveGrad)
        #self.query_grasp_prior_client = rospy.ServiceProxy(QUERY_GRASP_PRIOR_SERVICE_, GraspPriorGen)
        #self.VoxelInf = GraspVoxelInference()

        self.mdn = build_MDN()
        self.lkh = build_lkh()
        self.means = None

    def init_grasp_nets(self, sparse_grid, grid_dim, obj_size, batch_size=1):
        voxel = np.zeros(tuple(grid_dim))
        idx = np.reshape(sparse_grid, (-1,3)).astype(int)
        voxel[idx[:,0], idx[:,1], idx[:,2]] = 1
        grasp_obj = {}
        grasp_obj['voxel'] = voxel.reshape((1,1)+tuple(grid_dim))
        grasp_obj['object_dim'] = np.expand_dims(obj_size, 0)
        self.means, weights = self.mdn.object_condition_mdn(grasp_obj, batch_size)
        self.means = self.means[0].T
        for i in range(self.means.shape[0]):
            frame = 'mdn_mean_' + str(i+1)
            mean_pose = convert_array_to_pose(self.means[i][:6], 'object_pose')
            broadcast_tf(frame, mean_pose)
        self.lkh.object_condition(**grasp_obj)
        torch.cuda.empty_cache()

    def get_mean(self, idx):
        return self.means[idx]
        
    def Call_Service(self, grasp_obj):
        '''
        Client for grasp_voxel_inference on the segmented object.

        Params:
        grasp_obj - Output of point_cloud_segmentation (point_cloud_segmentation/GraspObject.msg)

        Output:
        Grasp_Pose - Pose of the palm link (6 dof float)
        Preshape - Preshape angle for fingers 1 & 2 (float)

        Status: Implemented

        Testing:
        '''
        warn_py2tf_call()
        if isinstance(grasp_obj, SegmentGraspObjectResponse):
            grasp_obj = grasp_obj.obj
        assert isinstance(grasp_obj, GraspObject), ("Param grasp_obj is not of type GraspObject,"
                                                    "Makesure it is the output of point_cloud_segmentation")
        vi_req_ = GraspVoxelInferRequest()
        vi_req_.seg_obj = grasp_obj
        vi_req_.prior_name = 'MDN'
        if (np.random.rand() < 0.5):
            vi_req_.grasp_type = 'overhead'
        else:
            vi_req_.grasp_type = 'side'
        vi_resp_ = self.Service(vi_req_)
        Grasp_Pose = vi_resp_.full_inf_config.palm_pose
        Preshape = vi_resp_.full_inf_config.hand_joint_state
        return Grasp_Pose, Preshape

    def query_grasp_probabilities(self, grasp_pose, preshape, sparse_grid, grid_dim, obj_size):
        #time0 = time.time()
        warn_py2tf_call()
        request = GraspObjectiveRequest()
        request.prior_name = "MDN"
        request.grasp_config = list(grasp_pose) + list(preshape)
        request.sparse_voxel_grid = tuple(sparse_grid.reshape(-1))
        request.voxel_grid_dim = grid_dim
        request.object_size = obj_size
        #rospy.loginfo(request.grasp_config)
        response = self.query_grasp_cost_client(request)
        #rospy.loginfo(f"Grasp cost took {time.time() - time0} secs")
        return response.posterior, response.likelihood, response.prior

    def query_grasp_grad(self, grasp_pose, preshape, sparse_grid, grid_dim, obj_size):
        warn_py2tf_call()
        request = GraspObjectiveGradRequest()
        request.prior_name = "MDN"
        request.grasp_config = list(grasp_pose) + list(preshape)
        request.sparse_voxel_grid = tuple(sparse_grid.reshape(-1))
        request.voxel_grid_dim = grid_dim
        request.object_size = obj_size
        response = self.query_grasp_grad_client(request)
        return response.posterior_grad, response.likelihood_grad, response.prior_grad

    def query_grasp_prior(self, sparse_grid, grid_dim, obj_size):
        '''
        Makes calls to prob_grasp_planner/grasp_voxel_inference to get MDN prior as initialization
        for the optimization.

        Params:
        sparse_grid - A sparse 3d occupancy grid of the object. (3d int)
        grid_dim - Dimension of the voxel grid.
        obj_size - Dimensions of the object to scale the voxel grid. 

        Output:
        grasp_init - 7 x 1 vector of grasp configuration. (float)
        
        Status: In Progress

        Testing:
        '''
        warn_py2tf_call()
        request = GraspPriorGenRequest()
        request.sparse_voxel_grid = tuple(sparse_grid.reshape(-1))
        request.voxel_grid_dim = grid_dim
        request.object_size = obj_size
        response = self.query_grasp_prior_client(request)
        return response.grasp_prior


    def get_grasp_probs(self, grasp_config, wt_post):
        grasp_config = torch.from_numpy(grasp_config).cuda()
        grasp_config = torch.unsqueeze(grasp_config, 0)
        lkh = -self.lkh.query_grasp_lkh(grasp_config).item() #returns ll
        prior = self.mdn.query_grasp_prior(grasp_config).item() #returns nll
        return wt_post[0] * lkh + wt_post[1] * prior, lkh, prior
    
    def get_grasp_prior(self, grasp_config):
        '''
        returns the -log(prior : grasp_config)
        '''
        grasp_config = torch.from_numpy(grasp_config).cuda()
        return self.mdn.query_grasp_prior(grasp_config).item()

    def test_comp_mixture(self, grasp_config):
        grasp_config = torch.from_numpy(grasp_config).cuda()
        prior, (comps, pis) = self.mdn.query_grasp_prior(grasp_config, comps=True)
        prior = prior
        #comps = -(comps - pis)
        return prior, -comps, -(comps - pis)


    def get_grasp_grads(self, grasp_config, wt_post):
        grasp_config = torch.from_numpy(grasp_config).cuda()
        grasp_config = torch.unsqueeze(grasp_config, 0)
        #print(grasp_config.shape)
        lkh_grad = -self.lkh.query_grasp_lkh_grad(grasp_config) #returns ll
        prior_grad = self.mdn.query_grasp_prior_grad(grasp_config) #returns nll
        return (wt_post[0] * lkh_grad + wt_post[1] * prior_grad)[0]
    
    def get_grasp_prior_grad(self, grasp_config):
        '''
        Params:
        grasp_config - current grasp_configuration to query at (7 x 1 float).

        Output:
        returns 1 x 7 of gradient -log(prior : grasp_config) w.r.t. grasp_config

        Status: In Progress

        Testing:
        '''
        grasp_config = torch.from_numpy(grasp_config).cuda()
        grad = self.mdn.query_grasp_prior_grad(grasp_config)
        return grad

    def get_grasp_lkh_grad(self, grasp_config):
        grasp_config = torch.from_numpy(grasp_config).cuda()
        grasp_config = torch.unsqueeze(grasp_config, 0)
        lkh_grad = -self.lkh.query_grasp_lkh_grad(grasp_config)
        return lkh_grad[0]

    def sample_prior(self):
        return self.mdn.sample().detach().cpu().numpy()[0]

    def sample_explicit(self, idx):
        return self.mdn.sample_explicit(idx).detach().cpu().numpy()[0]


    def get_torch_memory(self):
        mem = self.mdn.element_size() * self.mdn.nelement()
        mem += self.lkh.element_size() * self.lkh.nelement()
        return mem
