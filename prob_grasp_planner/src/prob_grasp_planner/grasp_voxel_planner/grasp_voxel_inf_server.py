#!/usr/bin/env python
import roslib
import rospy
from prob_grasp_planner.srv import *
from grasp_voxel_inference import GraspVoxelInference
import numpy as np
import time
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_common_library')
import prob_grasp_planner.grasp_common_library.grasp_common_functions as gcf
import grasp_pipeline.align_object_frame as align_obj
from prob_grasp_planner.grasp_common_library.data_proc_lib import DataProcLib 
import prob_grasp_planner.grasp_common_library.show_voxel
import tensorflow as tf
import copy


class GraspVoxelInfServer:


    def __init__(self):
        rospy.init_node('grasp_voxel_inf_server')
        end_effector = rospy.get_param('~end_effector', 'allegro')
        vis_preshape = rospy.get_param('~vis_preshape', False)
        virtual_hand_parent_tf = rospy.get_param('~virtual_hand_parent_tf', '')

        grasp_net_model_path = pkg_path + '/models/grasp_al_net/' + \
                           'grasp_net_freeze_enc_10_sets.ckpt'
        prior_model_path = pkg_path + '/models/grasp_al_prior/' + \
                            'prior_net_freeze_enc_10_sets.ckpt'
        gmm_model_path = pkg_path + '/models/grasp_al_prior/gmm_10_sets'

        rospy.loginfo("Configuring grasp_voxel_inf_server for %s", end_effector)

        if(end_effector=='reflex'):
            grasp_net_model_path = pkg_path + '/models/reflex_grasp_inf_models/grasp_voxel_inf_net/' + \
                           'grasp_voxel_net_reflex.ckpt'
            prior_model_path = pkg_path + '/models/reflex_grasp_inf_models/grasp_voxel_inf_prior/' + \
                               'prior_net_reflex.ckpt'
            gmm_model_path = pkg_path + '/models/reflex_grasp_inf_models/grasp_voxel_inf_prior/' + \
                             'failure_gmm_sets'
        # prior_model_path = pkg_path + '/models/grasp_al_prior/' + \
        #                     'suc_prior_net_freeze_enc_10_sets.ckpt'
        # gmm_model_path = pkg_path + '/models/grasp_al_prior/suc_gmm_10_sets'
        self.grasp_voxel_inf = GraspVoxelInference(grasp_net_model_path, 
                                        prior_model_path, gmm_model_path, 
                                        vis_preshape=vis_preshape,
                                        virtual_hand_parent_tf=virtual_hand_parent_tf,
                                        end_effector=end_effector) 

        self.cvtf = gcf.ConfigConvertFunctions(end_effector = end_effector)
        self.data_proc_lib = DataProcLib()


    def handle_grasp_voxel_inf(self, req):
        print 'prior:', req.prior_name

        obj_world_pose_stamp = PoseStamped()
        obj_world_pose_stamp.header.frame_id = req.seg_obj.header.frame_id
        obj_world_pose_stamp.pose = req.seg_obj.pose
        self.data_proc_lib.update_object_pose_client(obj_world_pose_stamp)

        config_init = None
        if req.prior_name == 'Constraint':
            init_hand_config = copy.deepcopy(req.init_hand_config)
            init_hand_config.palm_pose.header.stamp = rospy.Time.now() 
            init_hand_config.palm_pose = \
                    self.data_proc_lib.trans_pose_to_obj_tf(init_hand_config.palm_pose, 
                                                            return_array=False)
            config_init = self.cvtf.convert_full_to_preshape_config(init_hand_config)

        response = GraspVoxelInferResponse()

        # seg_obj_resp = gcf.segment_object_client(self.data_proc_lib.listener)
        # if not seg_obj_resp.object_found:
        #     response.success = False
        #     print 'No object found for segmentation!'
        #     return response 

        sparse_voxel_grid, voxel_size, voxel_grid_dim = self.data_proc_lib.voxel_gen_client(req.seg_obj)
        
        # show_voxel.plot_voxel(sparse_voxel_grid) 

        voxel_grid = np.zeros(tuple(voxel_grid_dim))
        voxel_grid_index = sparse_voxel_grid.astype(int)
        voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], 
                    voxel_grid_index[:, 2]] = 1
        voxel_grid = np.expand_dims(voxel_grid, -1)

        obj_size = np.array([req.seg_obj.width, req.seg_obj.height, 
                                    req.seg_obj.depth])

        comp_mean_config = self.grasp_voxel_inf.mdn_mean_explicit(voxel_grid, 
                                                    obj_size, req.grasp_type)
        print 'comp_mean_config: ', comp_mean_config
        full_comp_mean_config = self.cvtf.convert_preshape_to_full_config(
                                            comp_mean_config, 'object_pose')

        config_inf, obj_val_inf, config_init, obj_val_init, plan_info = \
            self.grasp_voxel_inf.max_grasp_suc_bfgs(voxel_grid, obj_size, 
                                    req.prior_name, config_init, req.grasp_type)

        full_config_init = self.cvtf.convert_preshape_to_full_config(
                                            config_init, 'object_pose')
        full_config_inf = self.cvtf.convert_preshape_to_full_config(
                                            config_inf, 'object_pose')

        self.grasp_voxel_inf.preshape_config = full_comp_mean_config
        # self.grasp_voxel_inf.preshape_config = full_config_inf
        # self.grasp_voxel_inf.pub_preshape_config()

        self.data_proc_lib.update_palm_pose_client(full_config_inf.palm_pose)

        response.inf_config_array = config_inf
        response.full_inf_config = full_config_inf
        response.inf_val = obj_val_inf
        response.inf_suc_prob = plan_info['inf_suc_prob']
        response.inf_log_prior = plan_info['inf_log_prior']
        response.init_val = obj_val_init
        response.init_suc_prob = plan_info['init_suc_prob']
        response.init_log_prior = plan_info['init_log_prior']
        response.init_config_array = config_init
        response.full_init_config = full_config_init
        response.sparse_voxel_grid = sparse_voxel_grid.flatten()
        response.voxel_size = voxel_size
        response.voxel_grid_dim = voxel_grid_dim
        response.object_size = obj_size
        response.success = True
        return response


    def create_voxel_inf_server(self):
        '''
            Create grasp inference server for the grasp model, 
            including the voxel classifier net and the MDN/GMM prior.
        '''
        rospy.Service('grasp_voxel_infer', GraspVoxelInfer, 
                        self.handle_grasp_voxel_inf)
        rospy.loginfo('Service grasp_voxel_infer:')
        rospy.loginfo('Ready to perform voxel grasp inference.')

    def handle_prior_generation(self, req):
        '''
        Configures the prior mixture and generates a grasp prior for the
        given object
        '''
        resp = GraspPriorGenResponse()
        voxel_grid = np.zeros(tuple(req.voxel_grid_dim))
        voxel_grid_index = np.reshape(req.sparse_voxel_grid,(-1,3)).astype(int)
        voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], 
                    voxel_grid_index[:, 2]] = 1
        voxel_grid = np.expand_dims(voxel_grid, -1)

        obj_size = np.array(req.object_size)
        self.grasp_voxel_inf.get_prior_mixture(voxel_grid, obj_size)
        resp.grasp_prior = self.grasp_voxel_inf.sample_mdn_implicit().reshape(-1).tolist()
        print(resp.grasp_prior)
        return resp
        
    def handle_grasp_log_posterior(self, req):
        '''
        Returns negative log posterior for the requested grasp and object
        '''
        resp = GraspObjectiveResponse()
        #grasp_config = self.cvtf.convert_full_to_preshape_config(req.full_grasp_config)

        voxel_grid = np.zeros(tuple(req.voxel_grid_dim))
        voxel_grid_index = np.reshape(req.sparse_voxel_grid,(-1,3)).astype(int)
        voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], 
                    voxel_grid_index[:, 2]] = 1
        voxel_grid = np.expand_dims(voxel_grid, -1)

        obj_size = np.array(req.object_size)

        resp.posterior, resp.likelihood, resp.prior = \
                                        self.grasp_voxel_inf.grasp_log_norm_posterior( \
                                            req.grasp_config, voxel_grid, obj_size, req.prior_name)

        return resp

    def handle_grasp_log_posterior_grad(self, req):
        '''
        Returns gradient of negative log posterior for the requested grasp and object
        '''
        resp = GraspObjectiveGradResponse()
        voxel_grid = np.zeros(tuple(req.voxel_grid_dim))
        voxel_grid_index = np.reshape(req.sparse_voxel_grid,(-1,3)).astype(int)
        voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], 
                    voxel_grid_index[:, 2]] = 1
        voxel_grid = np.expand_dims(voxel_grid, -1)

        obj_size = np.array(req.object_size)
        
        resp.posterior_grad, resp.likelihood_grad, resp.prior_grad = \
                                        self.grasp_voxel_inf.grasp_log_norm_posterior_grad(\
                                            req.grasp_config, voxel_grid, obj_size, req.prior_name)
        return resp

    def create_voxel_log_servers(self):
        '''
        Create ros servers to request grasp posteriors and posterior grads.
        '''
        rospy.Service('query_grasp_prior', GraspPriorGen, self.handle_prior_generation)
        rospy.Service('query_grasp_posterior', GraspObjective, self.handle_grasp_log_posterior)
        rospy.Service('query_grasp_posterior_grad', GraspObjectiveGrad, self.handle_grasp_log_posterior_grad)
        rospy.loginfo('Call service \"query_grasp_prior\" to set up and query from grasp MDN \n' \
                      + 'Note: This service needs to be called first befor calling the following services')
        rospy.loginfo('Call service \"query_grasp_posterior\" to query grasp cost')
        rospy.loginfo('Call service \"query_grasp_posterior_grad\" to query grasp cost gradients')

if __name__ == '__main__':
   grasp_voxel_inf_server = GraspVoxelInfServer() 
   grasp_voxel_inf_server.create_voxel_inf_server()
   grasp_voxel_inf_server.create_voxel_log_servers()
   rospy.spin()

