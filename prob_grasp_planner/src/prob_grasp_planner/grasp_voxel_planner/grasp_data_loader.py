import numpy as np
import h5py
import copy
import torch


class GraspDataLoader():
    # Reference: http://www.cvc.uab.es/people/joans/
    # slides_tensorflow/tensorflow_html/feeding_and_queues.html

    def __init__(self, data_path):
        self.data_path = data_path 
        self.voxel_grid_full_dim = [32, 32, 32]
        #For Allegro:
        #self.preshape_config_idx = list(xrange(8)) + [10, 11] + \
                                    #[14, 15] + [18, 19]
        #For Reflex:
        #self.preshape_config_idx = list(xrange(8)) + [8,9,10] + [8,9,10] #Depricated: Use this for non-tailored model
        self.preshape_config_idx = list(range(7))
        self.open_data_file()
        self.epochs_completed = 0
        self.shuffle(is_shuffle=False)
        self.index_in_epoch = 0
        self.starts_new_epoch = True
        print(f'Loaded data file: {self.data_path} with {self.num_samples} grasps')


    def open_data_file(self):
        data_file = h5py.File(self.data_path, 'r')
        self.num_samples = data_file['grasps_number'][()]
        # self.num_samples = 6630 
        data_file.close()


    def shuffle(self, is_shuffle=True):
        if is_shuffle:
            self.randperm = np.random.permutation(self.num_samples)
        else:
            self.randperm = list(range(self.num_samples))
 

    def next_batch(self, batch_size, 
                    is_shuffle=True, top_label=False, as_dict=True):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_samples:
            # Finished epoch
            self.starts_new_epoch = True
            self.epochs_completed += 1            
            self.shuffle(is_shuffle)
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_samples, f"Request {batch_size} but dataset has only {self.num_samples}"
        else:
            self.starts_new_epoch = False

        end = self.index_in_epoch        
        if not top_label:
            data = self.load_batches(start, end)
            if as_dict:
                batch= {}
                batch['grasp_config'] = data[0]
                batch['voxel'] = np.expand_dims(np.squeeze(data[1], -1), 1)
                batch['object_dim'] = data[2]
                batch['label'] = data[3]
                return batch
            return data
        else:
            return self.load_batches(start, end), \
                     self.load_batch_top_labels(start, end)


    def load_batches(self, start, end):
        data_file = h5py.File(self.data_path, 'r')
        grasp_configs = []
        grasp_voxel_grids = []
        grasp_obj_sizes = []
        grasp_labels = []
        for i in range(start, end):
            grasp_id = self.randperm[i]
            grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
            grasp_full_config = data_file[grasp_config_obj_key][()] 
            grasp_preshape_config = grasp_full_config[self.preshape_config_idx]
            grasp_sparse_voxel_key = 'grasp_' + str(grasp_id) + '_sparse_voxel'
            sparse_voxel_grid = data_file[grasp_sparse_voxel_key][()]
            obj_dim_key = 'grasp_' + str(grasp_id) + '_dim_w_h_d'
            obj_size = data_file[obj_dim_key][()]
            grasp_label_key = 'grasp_' + str(grasp_id) + '_label'
            grasp_label = data_file[grasp_label_key][()]

            voxel_grid = np.zeros(tuple(self.voxel_grid_full_dim))
            voxel_grid_index = sparse_voxel_grid.astype(int)
            voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1],
                        voxel_grid_index[:, 2]] = 1

            grasp_configs.append(grasp_preshape_config)
            grasp_voxel_grids.append(voxel_grid)
            grasp_obj_sizes.append(obj_size)
            grasp_labels.append(grasp_label)

        grasp_labels = np.expand_dims(grasp_labels, -1)
        grasp_voxel_grids = np.expand_dims(grasp_voxel_grids, -1)
        data_file.close()

        return grasp_configs, grasp_voxel_grids, \
                grasp_obj_sizes, grasp_labels


    def load_batch_top_labels(self, start, end):
        data_file = h5py.File(self.data_path, 'r')
        top_labels = []
        for i in range(start, end):
            grasp_id = self.randperm[i]
            top_label_key = 'grasp_' + str(grasp_id) + '_top_grasp'
            top_label = data_file[top_label_key][()]
            top_labels.append(top_label)

        top_labels = np.expand_dims(top_labels, -1)
        data_file.close()

        return top_labels


    def load_grasp_configs(self):
        data_file = h5py.File(self.data_path, 'r')
        grasp_configs = []
        for grasp_id in range(self.num_samples):
            grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
            grasp_full_config = data_file[grasp_config_obj_key][()] 
            grasp_preshape_config = grasp_full_config[self.preshape_config_idx]
            grasp_configs.append(grasp_preshape_config)

        data_file.close()

        return grasp_configs


    def display_obj_dims(self):
        c_count = 0
        data_file = h5py.File(self.data_path, 'r')
        for i in range(self.num_samples):
            grasp_id = i
            obj_dim_key = 'grasp_' + str(grasp_id) + '_dim_w_h_d'
            obj_size = data_file[obj_dim_key][()]
            if (obj_size[0] == obj_size[1]) or (obj_size[0] == obj_size[2]) \
               or (obj_size[1] == obj_size[2]):
                print('\033[93m' + str(obj_size) + '\033[0m')
                c_count +=1
            else: 
                print(obj_size)
        data_file.close()
        print(f'{c_count} of {self.num_samples} data corrupted ({100*c_count/self.num_samples}%)')

def convert_to_torch(sample, dtype=torch.float32, device = torch.device('cuda')):
    for key in sample:
        if type(sample[key]).__module__ == np.__name__:
            sample[key] = torch.from_numpy(sample[key]).type(dtype).to(device)
        elif type(sample[key]).__module__ == torch.__name__:
            sample[key] = sample[key].type(dtype).to(device)
        elif type(sample[key]) == list:
            sample[key] = torch.tensor(np.array(sample[key]), dtype=dtype, device=device)
        else:
            raise Exception("Value passed is neither numpy nor torch")
        sample[key] = sample[key].to(device)
    return sample

if __name__ == '__main__':
    train_data_path = '/home/mohanraj/reflex_grasp_data/processed/batch3/' + \
                        'grasp_voxelized_data.h5'
    gdl = GraspDataLoader(train_data_path)
    gdl.display_obj_dims()
