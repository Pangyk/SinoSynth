import h5py
import numpy as np
import os
import torch
from torch.utils import data
import SimpleITK as sitk
from ma_simul.ma_config import set_config_for_ct_artifact_simulation, set_config_for_cbct_artifact_simulation
from ma_simul.simul import add_ct_noise, add_cbct_noise

import warnings
warnings.filterwarnings('ignore')


def transform(x, y, mask):
    # selection = np.random.rand(2)
    #if selection[0] > 0.5:
    #    x = torch.flip(x, (0,))
    #    y = torch.flip(y, (0,))
    #    mask = torch.flip(mask, (0,))
    # if selection[1] > 0.5:
    #    x = torch.flip(x, (1,))
    #    y = torch.flip(y, (1,))
    #    mask = torch.flip(mask, (0,))

    return x, y, mask


def norm_data(x, mean, std):
    # attention here
    #x = x + 1000
    x = torch.clip(x, -1000, 1000)
    # x = torch.clip(x, -500, 2000)
    x = (x + 1000) / 2000
    # x = (x + 500) / 2500

    return x


class HDF5Dataset(data.Dataset):

    def __init__(self, root, args, is_transform=False):
        super().__init__()

        self.is_transform = is_transform
        self.mean = args.ct_mean
        self.std = args.ct_std
        self.args = args

        with h5py.File(root, "r") as f:
            dataset = np.array(f["data"], np.float32)

        self.data = dataset.reshape((-1, dataset.shape[2], dataset.shape[3]))
        self.config = set_config_for_ct_artifact_simulation(args.pixel_size)

    def __getitem__(self, index):
        # get data
        x = self.data[index]
        mask = np.zeros_like(x, dtype=np.float32)
        # mask[x >= 2400] = 1.0
        mask[x >= 1400] = 1.0
        mask = torch.from_numpy(mask)
        y = self.produce_y(x)

        x = norm_data(torch.from_numpy(x), self.mean, self.std)
        y = norm_data(torch.from_numpy(y), self.mean, self.std)
        if self.is_transform:
            x, y, mask = transform(x, y, mask)

        return x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0)

    def produce_y(self, x):
        min_v = [self.args.bone_s, self.args.water_s, self.args.m_s, self.args.n_s, self.args.p_s]
        max_v = [self.args.bone_e, self.args.water_e, self.args.m_e, self.args.n_e, self.args.p_e]
        vs = np.random.uniform(min_v, max_v, 5)
        self.config['bone_level'] = vs[0]
        self.config['water_level'] = vs[1]
        self.config['metal_density'] = vs[2]
        self.config['noise_scale'] = vs[3]
        self.config['percent_stripe'] = vs[4]

        y = add_ct_noise(x, self.config)
        return y

    def __len__(self):
        return len(self.data)


class CBCTDataset:
    def __init__(self, root, args):
        with h5py.File(root, "r") as f:
            self.dataset = np.array(f["data"], np.float32)

        self.data = self.dataset.reshape((-1, self.dataset.shape[2], self.dataset.shape[3]))
        self.non_empty_list = self.get_non_empty_list(self.data)
        self.args = args
        self.config = set_config_for_cbct_artifact_simulation(args.pixel_size)

    @staticmethod
    def get_non_empty_list(data):
        idx_list = []
        for i in range(len(data)):
            slc = data[i]
            region = np.count_nonzero(slc - slc.min())
            if region >= 500:
                idx_list.append(i)
        return idx_list

    def get_cbct_data(self):
        return self.dataset

    def get_samples(self, n):
        idx = np.random.randint(0, len(self.non_empty_list), n, dtype=np.int32)[0]
        slcs = self.data[self.non_empty_list[idx]]
        # y = add_cbct_noise(slcs, self.config)
        y = slcs

        slcs = norm_data(torch.from_numpy(slcs), 0, 0)
        y = norm_data(torch.from_numpy(y), 0, 0)
        return slcs.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0)

    def get_slice(self, pid, sid):
        x = self.dataset[pid, sid]
        if x.min() == x.max():
            x = norm_data(torch.from_numpy(x), 0, 0)
            return x.unsqueeze(0).unsqueeze(0), None

        y = add_cbct_noise(x, self.config)

        x = norm_data(torch.from_numpy(x), 0, 0)
        # y = x
        y = norm_data(torch.from_numpy(y), 0, 0)
        return x.unsqueeze(0).unsqueeze(0), x.unsqueeze(0).unsqueeze(0)

    def get_dim(self):
        p, s, _, _ = self.dataset.shape
        return p, s


def to_nii(x, idx, pid_file, output_path):
    with open(pid_file, "r") as f:
        pid_list = f.readlines()

    pid = pid_list[idx].replace("\n", "")
    x = x * 2000 - 1000
    img = sitk.GetImageFromArray(x)
    # img.SetSpacing((1, 1, 2))
    img.SetSpacing((1.65, 1.65, 3))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sitk.WriteImage(img, output_path + pid + ".nii.gz")

# if __name__ == "__main__":
#     with h5py.File(r'E:\workspace\research\SR\data\h5_cbct\testing_CBCT.h5', "r") as f:
#         dataset = np.array(f["data"], np.float32)
#     x = dataset[0][55]
#
#     x = np.clip(x, -500, 1000)
#     x = (x + 1000) / 2000
#
#     plt.imshow(x, cmap='gray')
#     plt.title('metal artifacts sinogram')
#     plt.show()
