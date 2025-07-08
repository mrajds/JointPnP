import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class VoxelAEDataset(Dataset):
    """Voxel AE dataset."""

    def __init__(self, root_dir, percentage = None):
        """
        Args:
            root_dir (string): Directory with all the files.
            percentage (float): Percentage of the files to take from root_dir
        """
        if not root_dir.endswith("/"):
            root_dir += "/"
        all_files = set(os.listdir(root_dir))
        self.files = []
        for f in all_files:
            if f.endswith(".npy"):
                no_extension_path = os.path.splitext(f)[0]
                no_extension_path = "-".join(no_extension_path.split("-")[:-1])
                if no_extension_path in all_files:
                    continue
                else:
                    self.files.append(root_dir + no_extension_path)

        if percentage is not None and percentage < 1.0:
            nr_files = int(percentage*len(self.files))
            self.files = random.sample(self.files, nr_files) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        full_path_no_extension = self.files[idx]
        splits = os.path.split(full_path_no_extension)
        root = splits[0] + "/"
        filename = splits[1]
        partial_view_path = root + filename + "-partial" + ".npy"
        full_view_path = root + filename + "-full" + ".npy"

        sample = {}

        with open(partial_view_path, "rb") as f:
            voxel = np.load(f) # 32 x 32 x 32
            voxel = torch.from_numpy(voxel.astype('float32'))
            voxel = voxel.unsqueeze(0) # 1 x 32 x 32 x 32
            sample["partial"] = voxel
        
        with open(full_view_path, "rb") as f:
            voxel = np.load(f)
            voxel = torch.from_numpy(voxel.astype('float32'))
            voxel = voxel.unsqueeze(0) # 1 x 32 x 32 x 32
            sample["full"] = voxel

        return sample

    def exclude_files(self, files_to_exclude):
        new_files = []
        for filename in self.files:
            if filename not in files_to_exclude:
                new_files.append(filename)
        self.files = new_files


def main():
    dataset = VoxelAEDataset("/mnt/Voxel_CF_numpy/Train/")
    print(len(dataset))
    for i, sample in enumerate(dataset):
        print(sample)
        if i == 10:
            break

if __name__ == "__main__":
    main()
