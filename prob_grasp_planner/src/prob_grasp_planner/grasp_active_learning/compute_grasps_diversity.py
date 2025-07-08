import numpy as np
import os
import h5py
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import mixture


def read_exp_data(data_path):
    grasp_data = []
    grasp_data_file = h5py.File(data_path, 'r')
    for i in xrange(5):
        for j in xrange(3):
            obj_grasp_id = 'object_' + str(i) + '_grasp_' + \
                    str(j) + '_inf_config_array'
            grasp = grasp_data_file[obj_grasp_id][()]
            grasp_data.append(grasp)
    grasp_data_file.close()
    return grasp_data


def compute_gmm_entropy(gmm):
    # I am not sure if this gmm entropy is correct.
    k = 14
    gmm_entropy = 0.
    for i, w in enumerate(gmm.weights_):
        print 'GMM dim: ', gmm.covariances_[i].shape
        entropy = 0.5 * (k + k * np.log(2 * np.pi) + \
                            np.log(np.linalg.det(gmm.covariances_[i]))) 
        gmm_entropy += w * entropy
    return gmm_entropy 


def compute_kernel_det(X, kernel_name='rbf'):
    # https://arxiv.org/pdf/1807.01477.pdf
    kernel_mat = None
    if kernel_name == 'linear':
        kernel_mat = metrics.pairwise.linear_kernel(X, Y=None)
    elif kernel_name == 'rbf':
        kernel_mat = metrics.pairwise.rbf_kernel(X, Y=None, gamma=None)
    elif kernel_name == 'laplacian':
        kernel_mat = metrics.pairwise.laplacian_kernel(X, Y=None, gamma=None)
    print 'kernel_mat: ', kernel_mat.shape
    kernel_det = np.linalg.det(kernel_mat)
    return kernel_det


def compute_data_limit_norm(d):
    print 'Data shape for limit norm: ', d.shape
    rng = np.max(d, axis=0) - np.min(d, axis=0)
    print rng
    n = np.linalg.norm(rng, ord=1)
    return n


def compute_exp_covariance(datasets_path):
    data_folders = os.listdir(datasets_path)

    grasp_data = []
    for data_folder in data_folders:
        data_path = datasets_path + data_folder
        print data_path
        grasp_one_obj = read_exp_data(data_path)
        grasp_data += grasp_one_obj
        print len(grasp_data)

    # print grasp_data

    active_data = np.array(grasp_data[::3]).T
    more_sup_data = np.array(grasp_data[1::3]).T
    # less_sup_data = np.array(grasp_data[2::3]).T
    # The first grasp of supervised learning with less data is 
    # empty, seems the data recording missed that grasp
    # active_data = np.array(grasp_data[3::3]).T
    # more_sup_data = np.array(grasp_data[4::3]).T
    less_sup_data = np.array(grasp_data[5::3]).T
    #print less_sup_data.shape

    # active_data = active_data[:6, :]
    # more_sup_data = more_sup_data[:6, :]
    # less_sup_data = less_sup_data[:6, :]

    # active_data = active_data[6:, :]
    # more_sup_data = more_sup_data[6:, :]
    # less_sup_data = less_sup_data[6:, :]

    active_cov = np.cov(active_data)
    more_cov = np.cov(more_sup_data)
    less_cov = np.cov(less_sup_data)

    # print active_cov
    # print more_cov
    # print less_cov
    # print active_cov.shape, more_cov.shape, less_cov.shape
    active_cov_det = np.linalg.det(active_cov)
    more_cov_det = np.linalg.det(more_cov)
    less_cov_det = np.linalg.det(less_cov)
    print 'active exp cov det: ', active_cov_det
    print 'sup more exp cov det: ', more_cov_det
    print 'sup less exp cov det: ', less_cov_det 

    k = active_data.shape[0]
    print 'k: ', k
    active_entropy = 0.5 * (k + k * np.log(2 * np.pi) + np.log(active_cov_det)) 
    more_entropy = 0.5 * (k + k * np.log(2 * np.pi) + np.log(more_cov_det)) 
    less_entropy = 0.5 * (k + k * np.log(2 * np.pi) + np.log(less_cov_det)) 

    print 'active entropy: ', active_entropy
    print 'more entropy: ', more_entropy
    print 'less entropy: ', less_entropy

    active_eigval, active_eigvec = np.linalg.eig(active_cov)
    more_eigval, more_eigvec = np.linalg.eig(more_cov)
    less_eigval, less_eigvec = np.linalg.eig(less_cov)

    # print active_eigval
    # print more_eigval
    # print less_eigval

    # print np.prod(active_eigval)

    num_comp = 2
    active_gmm = mixture.GaussianMixture(
                            n_components=num_comp, 
                            covariance_type='full', random_state=0, 
                            init_params='kmeans', n_init=5)
    active_gmm.fit(active_data.T)
    active_gmm_entropy = compute_gmm_entropy(active_gmm)

    more_sup_gmm = mixture.GaussianMixture(
                            n_components=num_comp, 
                            covariance_type='full', random_state=0, 
                            init_params='kmeans', n_init=5)
    more_sup_gmm.fit(more_sup_data.T)
    more_sup_gmm_entropy = compute_gmm_entropy(more_sup_gmm)

    less_sup_gmm = mixture.GaussianMixture(
                            n_components=num_comp, 
                            covariance_type='full', random_state=0, 
                            init_params='kmeans', n_init=5)
    less_sup_gmm.fit(less_sup_data.T)
    less_sup_gmm_entropy = compute_gmm_entropy(less_sup_gmm)

    print 'active_gmm_entropy: ', active_gmm_entropy
    print 'more_sup_gmm_entropy: ', more_sup_gmm_entropy
    print 'less_sup_gmm_entropy: ', less_sup_gmm_entropy

    active_kernel_det = compute_kernel_det(active_data.T)
    more_sup_kernel_det = compute_kernel_det(more_sup_data.T) 
    less_sup_kernel_det = compute_kernel_det(less_sup_data.T) 
    print 'active_kernel_det: ', active_kernel_det
    print 'more_sup_kernel_det: ', more_sup_kernel_det
    print 'less_sup_kernel_det: ', less_sup_kernel_det

    active_limit_norm = compute_data_limit_norm(active_data.T)
    more_sup_limit_norm = compute_data_limit_norm(more_sup_data.T)
    less_sup_limit_norm = compute_data_limit_norm(less_sup_data.T)
    print 'active_limit_norm: ', active_limit_norm
    print 'more_sup_limit_norm: ', more_sup_limit_norm
    print 'less_sup_limit_norm: ', less_sup_limit_norm

    plot_gaus = False
    if plot_gaus:
        plt.figure()
        plt.matshow(active_cov)
        fig_name = '/home/qingkai/diversity_vis/active_exp_cov.png' 
        plt.savefig(fig_name)
        plt.close()

        plt.figure()
        plt.matshow(more_cov)
        fig_name = '/home/qingkai/diversity_vis/more_exp_cov.png' 
        plt.savefig(fig_name)
        plt.close()

        plt.figure()
        plt.matshow(less_cov)
        fig_name = '/home/qingkai/diversity_vis/less_exp_cov.png' 
        plt.savefig(fig_name)
        plt.close()

        # Reference: https://stats.stackexchange.com/questions/12819/
        # how-to-draw-a-scree-plot-in-python 
        fig = plt.figure()
        sing_vals = np.arange(k) + 1
        plt.plot(sing_vals, np.sort(active_eigval)[::-1], 'r+-', linewidth=2)
        plt.plot(sing_vals, np.sort(more_eigval)[::-1], 'go--', linewidth=2)
        plt.plot(sing_vals, np.sort(less_eigval)[::-1], 'kx:', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        leg = plt.legend(['Active learning', 'Supervised more data', 
                        'Supervised less data'], loc='best', borderpad=0.3, 
                        shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                        markerscale=0.)
        fig_name = '/home/qingkai/diversity_vis/scree.png' 
        plt.savefig(fig_name)
        plt.close()

        fig = plt.figure()
        sing_vals = np.arange(k) + 1
        plt.plot(sing_vals, np.sort(np.log(active_eigval))[::-1], 'r+-', linewidth=2)
        plt.plot(sing_vals, np.sort(np.log(more_eigval))[::-1], 'go--', linewidth=2)
        plt.plot(sing_vals, np.sort(np.log(less_eigval))[::-1], 'kx:', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Log Eigenvalue')
        leg = plt.legend(['Active learning', 'Supervised more data', 
                        'Supervised less data'], loc='best', borderpad=0.3, 
                        shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                        markerscale=0.)
        fig_name = '/home/qingkai/diversity_vis/scree_log.png' 
        plt.savefig(fig_name)
        plt.close()


def compute_active_learn_cov(active_data_path):
    active_data_file = h5py.File(active_data_path, 'r')
    num_act_samples = active_data_file['total_grasps_num'][()]
    print 'num_act_samples: ', num_act_samples
    grasp_configs_raw = []
    for grasp_id in xrange(num_act_samples):
        obj_grasp_id_key = 'grasp_' + str(grasp_id) + '_obj_grasp_id'
        object_grasp_id = active_data_file[obj_grasp_id_key][()]
        inf_config_array_key = object_grasp_id + '_inf_config_array'
        grasp_preshape_config = active_data_file[inf_config_array_key][()]
        grasp_configs_raw.append(grasp_preshape_config)
    active_data_file.close()

    # grasp_configs_raw = np.array(grasp_configs_raw)[:, 6:]
    
    grasp_configs = np.array(grasp_configs_raw).T
    data_cov = np.cov(grasp_configs)
    cov_det = np.linalg.det(data_cov)
    cov_eigval, cov_eigvec = np.linalg.eig(data_cov)
    print 'active learning cov det:', cov_det
    # print cov_eigval

    k = grasp_configs.shape[0]
    print 'k: ', k
    active_learn_entropy = 0.5 * (k + k * np.log(2 * np.pi) + np.log(cov_det)) 
    print 'active learning entropy: ', active_learn_entropy

    num_comp = 2
    active_learn_gmm = mixture.GaussianMixture(
                            n_components=num_comp, 
                            covariance_type='full', random_state=0, 
                            init_params='kmeans', n_init=5)
    active_learn_gmm.fit(grasp_configs_raw)
    active_learn_gmm_entropy = compute_gmm_entropy(active_learn_gmm)
    print 'active_learn_gmm_entropy: ', active_learn_gmm_entropy

    active_learn_kernel_det = compute_kernel_det(grasp_configs_raw)
    print 'active_learn_kernel_det: ', active_learn_kernel_det

    active_learn_limit_norm = compute_data_limit_norm(np.array(grasp_configs_raw))
    print 'active_learn_limit_norm: ', active_learn_limit_norm

    return np.sort(cov_eigval)[::-1]


def compute_heu_cov(heu_data_path):
    grasp_configs = []

    heu_data_file = h5py.File(heu_data_path, 'r')
    num_heu_samples = heu_data_file['grasps_number'][()]
    preshape_config_idx = list(xrange(8)) + [10, 11] + \
                                    [14, 15] + [18, 19]

    for grasp_id in xrange(num_heu_samples):
        grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
        grasp_full_config = heu_data_file[grasp_config_obj_key][()] 
        grasp_preshape_config = grasp_full_config[preshape_config_idx]
        grasp_configs.append(grasp_preshape_config)
    heu_data_file.close()

    num_act_samples = 2052
    randperm = np.random.permutation(num_heu_samples)
    i = num_act_samples
    cov_det_subsets = []
    eigval_subsets = []
    entropy_subsets = []
    num_comp = 2
    gmm_entropy_subsets = []
    kernel_det_subsets = []
    limit_norm_subsets = []
    while i < num_heu_samples:
        print 'heu subset: ', i
        grasp_data_subset = np.array(grasp_configs[i-num_act_samples:i]).T
        # grasp_data_subset = grasp_data_subset[6:, :]
        data_cov = np.cov(grasp_data_subset)
        cov_det = np.linalg.det(data_cov)
        cov_eigval, cov_eigvec = np.linalg.eig(data_cov)
        print 'cov det', cov_det
        # print cov_eigval
        k = grasp_data_subset.shape[0]
        print 'k: ', k
        heu_entropy = 0.5 * (k + k * np.log(2 * np.pi) + np.log(cov_det)) 
        print 'heuristic entropy: ', heu_entropy
        i += num_act_samples
        cov_det_subsets.append(cov_det)
        eigval_subsets.append(np.sort(cov_eigval)[::-1])
        entropy_subsets.append(heu_entropy)

        heu_gmm = mixture.GaussianMixture(
                            n_components=num_comp, 
                            covariance_type='full', random_state=0, 
                            init_params='kmeans', n_init=5)
        heu_gmm.fit(grasp_data_subset.T)
        heu_gmm_entropy = compute_gmm_entropy(heu_gmm)
        gmm_entropy_subsets.append(heu_gmm_entropy)

        heu_kernel_det = compute_kernel_det(grasp_data_subset.T)
        # print 'heu_kernel_det: ', heu_kernel_det
        kernel_det_subsets.append(heu_kernel_det)

        heu_limit_norm = compute_data_limit_norm(grasp_data_subset.T)
        limit_norm_subsets.append(heu_limit_norm)

    print 'mean cov det: ', np.mean(cov_det_subsets)
    print 'std cov det: ', np.std(cov_det_subsets)

    print 'mean entropy: ', np.mean(entropy_subsets)
    print 'std entropy: ', np.std(entropy_subsets)

    eigval_mean = np.mean(eigval_subsets, axis=0)
    eigval_std = np.std(eigval_subsets, axis=0)

    heu_gmm_mean = np.mean(gmm_entropy_subsets)
    heu_gmm_std = np.std(gmm_entropy_subsets)
    print 'heu_gmm_mean: ', heu_gmm_mean
    print 'heu_gmm_std: ', heu_gmm_std

    heu_kernel_det_mean = np.mean(kernel_det_subsets)
    heu_kernel_det_std = np.std(kernel_det_subsets)
    print 'heu_kernel_det_mean: ', heu_kernel_det_mean
    print 'heu_kernel_det_std: ', heu_kernel_det_std

    heu_limit_norm_mean = np.mean(limit_norm_subsets)
    heu_limit_norm_std = np.std(limit_norm_subsets)
    print 'heu_limit_norm_mean: ', heu_limit_norm_mean
    print 'heu_limit_norm_std: ', heu_limit_norm_std


    return eigval_mean, eigval_std


def joint_angle_limits():
    index_joint_0_lower = -0.59
    index_joint_0_upper = 0.57
    middle_joint_0_lower = -0.59
    middle_joint_0_upper = 0.57
    ring_joint_0_lower = -0.59
    ring_joint_0_upper = 0.57

    index_joint_1_lower = -0.296
    index_joint_1_upper = 0.71
    middle_joint_1_lower = -0.296
    middle_joint_1_upper = 0.71
    ring_joint_1_lower = -0.296
    ring_joint_1_upper = 0.71

    thumb_joint_0_lower = 0.363
    thumb_joint_0_upper = 1.55
    thumb_joint_1_lower = -0.205
    thumb_joint_1_upper = 1.263

    index_joint_0_middle = (index_joint_0_lower + index_joint_0_upper) * 0.5
    middle_joint_0_middle = (middle_joint_0_lower + middle_joint_0_upper) * 0.5
    ring_joint_0_middle = (ring_joint_0_lower + ring_joint_0_upper) * 0.5
    index_joint_1_middle = (index_joint_1_lower + index_joint_1_upper) * 0.5
    middle_joint_1_middle = (middle_joint_1_lower + middle_joint_1_upper) * 0.5
    ring_joint_1_middle = (ring_joint_1_lower + ring_joint_1_upper) * 0.5
    thumb_joint_0_middle = (thumb_joint_0_lower + thumb_joint_0_upper) * 0.5
    thumb_joint_1_middle = (thumb_joint_1_lower + thumb_joint_1_upper) * 0.5

    index_joint_0_range = index_joint_0_upper - index_joint_0_lower
    middle_joint_0_range = middle_joint_0_upper - middle_joint_0_lower
    ring_joint_0_range = ring_joint_0_upper - ring_joint_0_lower
    index_joint_1_range = index_joint_1_upper - index_joint_1_lower
    middle_joint_1_range = middle_joint_1_upper - middle_joint_1_lower
    ring_joint_1_range = ring_joint_1_upper - ring_joint_1_lower
    thumb_joint_0_range = thumb_joint_0_upper - thumb_joint_0_lower
    thumb_joint_1_range = thumb_joint_1_upper - thumb_joint_1_lower

    first_joint_lower_limit = 0.25
    first_joint_upper_limit = 0.25
    second_joint_lower_limit = 0.5
    second_joint_upper_limit = 0. #-0.1

    thumb_1st_joint_lower_limit = -0.5
    thumb_1st_joint_upper_limit = 0.5
    thumb_2nd_joint_lower_limit = 0.25
    thumb_2nd_joint_upper_limit = 0.25

    index_joint_0_sample_lower = index_joint_0_middle - first_joint_lower_limit * index_joint_0_range
    index_joint_0_sample_upper = index_joint_0_middle + first_joint_upper_limit * index_joint_0_range
    middle_joint_0_sample_lower = middle_joint_0_middle - first_joint_lower_limit * middle_joint_0_range
    middle_joint_0_sample_upper = middle_joint_0_middle + first_joint_upper_limit * middle_joint_0_range
    ring_joint_0_sample_lower = ring_joint_0_middle - first_joint_lower_limit * ring_joint_0_range
    ring_joint_0_sample_upper = ring_joint_0_middle + first_joint_upper_limit * ring_joint_0_range

    index_joint_1_sample_lower = index_joint_1_middle - second_joint_lower_limit * index_joint_1_range
    index_joint_1_sample_upper = index_joint_1_middle + second_joint_upper_limit * index_joint_1_range
    middle_joint_1_sample_lower = middle_joint_1_middle - second_joint_lower_limit * middle_joint_1_range
    middle_joint_1_sample_upper = middle_joint_1_middle + second_joint_upper_limit * middle_joint_1_range
    ring_joint_1_sample_lower = ring_joint_1_middle - second_joint_lower_limit * ring_joint_1_range
    ring_joint_1_sample_upper = ring_joint_1_middle + second_joint_upper_limit * ring_joint_1_range

    thumb_joint_0_sample_lower = thumb_joint_0_middle - thumb_1st_joint_lower_limit * thumb_joint_0_range
    thumb_joint_0_sample_upper = thumb_joint_0_middle + thumb_1st_joint_upper_limit * thumb_joint_0_range
    thumb_joint_1_sample_lower = thumb_joint_1_middle - thumb_2nd_joint_lower_limit * thumb_joint_1_range
    thumb_joint_1_sample_upper = thumb_joint_1_middle + thumb_2nd_joint_upper_limit * thumb_joint_1_range

    sample_lower_limit = [index_joint_0_sample_lower, index_joint_1_sample_lower,
                        middle_joint_0_sample_lower, middle_joint_1_sample_lower,
                        ring_joint_0_sample_lower, ring_joint_1_sample_lower,
                        thumb_joint_0_sample_lower, thumb_joint_1_sample_lower]
    sample_upper_limit = [index_joint_0_sample_upper, index_joint_1_sample_upper,
                        middle_joint_0_sample_upper, middle_joint_1_sample_upper,
                        ring_joint_0_sample_upper, ring_joint_1_sample_upper,
                        thumb_joint_0_sample_upper, thumb_joint_1_sample_upper]
    return np.array(sample_lower_limit), np.array(sample_upper_limit)


def active_learn_grasps_low_heu_prob(active_data_path):
    lower_joint_limit, upper_joint_limit = joint_angle_limits()
    lower_joint_limit = np.delete(lower_joint_limit, -2)
    upper_joint_limit = np.delete(upper_joint_limit, -2)
    active_data_file = h5py.File(active_data_path, 'r')
    num_act_samples = active_data_file['total_grasps_num'][()]
    print 'num_act_samples: ', num_act_samples
    # grasp_configs_raw = []
    out_limit_val_dict = {}
    for grasp_id in xrange(num_act_samples):
        obj_grasp_id_key = 'grasp_' + str(grasp_id) + '_obj_grasp_id'
        object_grasp_id = active_data_file[obj_grasp_id_key][()]
        inf_config_array_key = object_grasp_id + '_inf_config_array'
        grasp_preshape_config = active_data_file[inf_config_array_key][()]
        cfg_joint_angles = grasp_preshape_config[6:]
        cfg_joint_angles = np.delete(cfg_joint_angles, -2)
        if np.any(cfg_joint_angles <= lower_joint_limit) or \
                np.any(cfg_joint_angles >= upper_joint_limit):
            clip_joint_angles = np.clip(cfg_joint_angles, lower_joint_limit, 
                                        upper_joint_limit)
            out_limit_val = np.linalg.norm((cfg_joint_angles 
                                            - clip_joint_angles), ord=1)
            out_limit_val_dict[object_grasp_id] = out_limit_val
            if object_grasp_id in ['object_207_grasp_4', 'object_161_grasp_4',
                    'object_259_grasp_2', 'object_441_grasp_1', 'object_91_grasp_3']:
                print cfg_joint_angles
                print lower_joint_limit
                print upper_joint_limit
                print 'grasp_id: ', grasp_id
                print 'object_grasp_id: ', object_grasp_id
        # grasp_configs_raw.append(grasp_preshape_config)
    active_data_file.close()
    sorted_dict = sorted(out_limit_val_dict.items(), 
                        key=lambda x: x[1], reverse=True)
    for i, (k, v) in enumerate(sorted_dict):
        if i > 10:
            break
        print k, v

    print 'Number of active learning grasps out of ' \
            'heu joint angle limits:', len(sorted_dict)


def exp_grasps_low_heu_prob(datasets_path):
    lower_joint_limit, upper_joint_limit = joint_angle_limits()
    lower_joint_limit = np.delete(lower_joint_limit, -2)
    upper_joint_limit = np.delete(upper_joint_limit, -2)

    data_folders = os.listdir(datasets_path)

    grasp_data = []
    for data_folder in data_folders:
        data_path = datasets_path + data_folder
        print data_path
        grasp_one_obj = read_exp_data(data_path)
        grasp_data += grasp_one_obj
        print len(grasp_data)

    active_data = np.array(grasp_data[::3])
    more_sup_data = np.array(grasp_data[1::3])
    less_sup_data = np.array(grasp_data[5::3])

    active_out_limits_num = 0
    for active_cfg in active_data:
        cfg_joint_angles = active_cfg[6:]
        cfg_joint_angles = np.delete(cfg_joint_angles, -2)
        if np.any(cfg_joint_angles <= lower_joint_limit) or \
                np.any(cfg_joint_angles >= upper_joint_limit):
                    active_out_limits_num += 1

    more_sup_out_limits_num = 0
    for more_sup_cfg in more_sup_data:
        cfg_joint_angles = more_sup_cfg[6:]
        cfg_joint_angles = np.delete(cfg_joint_angles, -2)
        if np.any(cfg_joint_angles <= lower_joint_limit) or \
                np.any(cfg_joint_angles >= upper_joint_limit):
                    more_sup_out_limits_num += 1

    less_sup_out_limits_num = 0
    for less_sup_cfg in less_sup_data:
        cfg_joint_angles = less_sup_cfg[6:]
        cfg_joint_angles = np.delete(cfg_joint_angles, -2)
        if np.any(cfg_joint_angles <= lower_joint_limit) or \
                np.any(cfg_joint_angles >= upper_joint_limit):
                    less_sup_out_limits_num += 1

    print 'active_out_limits_num: ', active_out_limits_num
    print 'more_sup_out_limits_num: ', more_sup_out_limits_num
    print 'less_sup_out_limits_num: ', less_sup_out_limits_num


def diversity_analysis():
    datasets_path = '/mnt/tars_data/active_exp_h5_data/' 
    compute_exp_covariance(datasets_path)
    print '################################'

    active_learn_data_path = '/mnt/tars_data/active_learn_h5_data/' + \
                            'active_grasp_data.h5'
    active_learn_eigval = compute_active_learn_cov(active_learn_data_path)
    print '------------------------------'

    heu_data_path = '/mnt/tars_data/active_learn_h5_data/' + \
                            'heu_grasp_data_10_sets.h5'
    heu_eigval_mean, heu_eigval_std = compute_heu_cov(heu_data_path)

    plot_gaus = False
    if plot_gaus:
        k = 14
        fig = plt.figure()
        sing_vals = np.arange(k) + 1
        plt.plot(sing_vals, active_learn_eigval, 'r+-', linewidth=1, markersize=1.5)
        plt.plot(sing_vals, heu_eigval_mean, 'go--', linewidth=1, markersize=1.5)
        plt.errorbar(sing_vals, heu_eigval_mean, heu_eigval_std, 
                    linestyle='None', color='g', capsize=1)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        leg = plt.legend(['Active learning', 'Heuristic'], loc='best', borderpad=0.3, 
                        shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                        markerscale=0.)
        fig_name = '/home/qingkai/diversity_vis/scree_learn.png' 
        plt.savefig(fig_name)
        plt.close()

        fig = plt.figure()
        sing_vals = np.arange(k) + 1
        plt.plot(sing_vals, np.log(active_learn_eigval), 'r+-', linewidth=1, markersize=1.5)
        plt.plot(sing_vals, np.log(heu_eigval_mean), 'go--', linewidth=1, markersize=1.5)
        plt.errorbar(sing_vals, np.log(heu_eigval_mean), heu_eigval_std, 
                    linestyle='None', color='g', capsize=1)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Log Eigenvalue')
        leg = plt.legend(['Active learning', 'Heuristic'], loc='best', borderpad=0.3, 
                        shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                        markerscale=0.)
        fig_name = '/home/qingkai/diversity_vis/scree_log_learn.png' 
        plt.savefig(fig_name)
        plt.close()


if __name__ == '__main__':
    diversity_analysis()
    # active_learn_data_path = '/mnt/tars_data/active_learn_h5_data/' + \
    #                         'active_grasp_data.h5'
    # active_learn_grasps_low_heu_prob(active_learn_data_path)
    # datasets_path = '/mnt/tars_data/active_exp_h5_data/' 
    # exp_grasps_low_heu_prob(datasets_path)

