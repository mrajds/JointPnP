#!/usr/bin/env python3

import rospy
import numpy as np
from ll4ma_pick_n_place.opt_problem import gen_query_grid_pts
from ll4ma_pick_n_place.data_utils import load_env_yaml
from ll4ma_pick_n_place.visualizations import map_rgb
from matplotlib import pyplot, colors
from gpu_sd_map.transforms_lib import vector2mat, transform_points


if __name__ == '__main__':
    rospy.init_node('sdf_viz_node')
    ENV_YAML = '/home/mohanraj/robot_WS/src/ll4ma_pick_n_place/envs/corner6_b.yaml'#rospy.get_param('env_yaml', None)
    Env_List = load_env_yaml(ENV_YAML)
    ex_obj = Env_List['pitcher']
    ex_obj.set_pose([0, 0, 0, 0, 0, 0], preserve=True)
    obj_size = ex_obj.true_size
    kernel_size = ex_obj.max_tsdf()*2e-3 + np.max(obj_size[:2]) + 0.1
    g = gen_query_grid_pts(kernel_size, kernel_size, 1000)
    #persp_trans = vector2mat([kernel_size*0.5*np.sin(0.785), -kernel_size*0.5*np.cos(0.785), -kernel_size/2], [0, -1.57, 0.785]) @ vector2mat(rpy=[0.0, -0.78, 0.0])
    #gt = transform_points(g, persp_trans)
    gt = g
    q = gt - np.array([kernel_size, kernel_size, 0])/2
    min_sd, dsd = ex_obj.query_points(q)
    for z in np.arange(-obj_size[2]/2, obj_size[2]/2, 0.001):
        #qn = z*persp_trans[:3,2] + q
        q[:,2] = z
        sd, dsd = ex_obj.query_points(q)
        min_sd = np.minimum(sd, min_sd)
    #print(min_sd)
    I = np.zeros(np.array([kernel_size*1e3+1, kernel_size*1e3+1]).astype(int))
    #I += 100
    ids = (np.around(g,3)*1e3).astype(int)
    B = np.zeros(I.shape)
    if min_sd is not None:
        I[ids[:,0], ids[:,1]] = min_sd
    B[I>0] = 1
    B[I<0] = 0

    fig, axes = pyplot.subplots(2, 1)
    c1 = axes[0].contourf(I, np.arange(0, max(min_sd), 10), cmap='winter')
    #norm = colors.SymLogNorm(vmin=min(min_sd), vmax=max(min_sd), linthresh=1, base=2)
    axes[0].clear()
    axes[0].imshow(B, cmap='gray')
    axes[0].contour(I, np.arange(0, max(min_sd), 10), cmap='winter')
    axes[0].xaxis.set_tick_params(labelbottom=False)
    axes[0].yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    fig.colorbar(c1)
    axes[0].set_aspect(1)

    c2 = axes[1].contourf(I, np.arange(min(min_sd), 0, 10), cmap='winter')
    axes[1].clear()
    axes[1].imshow(1-B, cmap='gray')
    axes[1].contour(I, np.arange(min(min_sd), 0, 10), cmap='winter')
    fig.colorbar(c2)
    axes[1].set_aspect(1)
    #fig.set_figheight(300)
    #fig.set_figwidth(300)

    axes[1].xaxis.set_tick_params(labelbottom=False)
    axes[1].yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    pyplot.show()

    
    
    #norm = colors.SymLogNorm(vmin=min(min_sd), vmax=max(min_sd), linthresh=0.1, base=1.01)
    norm = colors.BoundaryNorm(boundaries=np.array([-0.5, 0.5]), ncolors=ex_obj.max_tsdf()*2)
    pyplot.contourf(I, np.arange(min(min_sd), max(min_sd)+5, 0.5), cmap='rainbow', norm=norm)
    pyplot.gca().set_aspect(1)
    pyplot.show()
