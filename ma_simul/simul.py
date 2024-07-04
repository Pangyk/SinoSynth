import argparse

from ma_simul.utils import hu2mu, ct_metal_artifact_simulation
import h5py
import matplotlib.pyplot as plt
import numpy as np


max_num = 15


def recon_image(x):
    return np.clip(x, -500, 2500)


def add_mask(x, mask, min_m=0, max_m=6):
    ms = np.zeros_like(x)

    x_axis = np.sum(mask, axis=0)
    x_ids = np.nonzero(x_axis)[0]
    y_axis = np.sum(mask, axis=1)
    y_ids = np.nonzero(y_axis)[0]
    if not (len(x_ids) < 2 or len(y_ids) < 2):
        num = np.random.randint(min_m, max_m, size=1)[0]
        x_min, x_max, y_min, y_max = np.min(x_ids), np.max(x_ids), np.min(y_ids), np.max(y_ids)
        if not (x_min == x_max or y_min == y_max) and num > 0:
            xs = np.random.randint(x_min, x_max, size=(num,), dtype=np.int32)
            ys = np.random.randint(y_min, y_max, size=(num,), dtype=np.int32)
            r = int(max(x_max - x_min, y_max - y_min) / 5)
            # min_r = min(50, r)
            max_r = max(5, r)
            rs = np.random.randint(1, max_r, size=(num,), dtype=np.int32)
            for i in range(num):
                ms[(ys[i] - rs[i]):(ys[i] + rs[i]), (xs[i] - rs[i]):(xs[i] + rs[i])] = 1

    return ms


def add_ct_noise(x, config):
    image = recon_image(x)
    mask = np.zeros_like(image)
    mask[image > config['T3']] = 1

    ms = add_mask(image, mask, min_m=1, max_m=max_num)

    metal = np.zeros_like(image)
    metal[(mask > 0) & (ms > 0)] = 1

    # phantom = create_phantom(256, 256, 100, config['mu_water'])
    # config['correction_coeff'] = water_correction(phantom, config)
    # print(config['correction_coeff'])
    image_hu = hu2mu(image, config['mu_water'], config['mu_air'])

    x_t = ct_metal_artifact_simulation(image_hu, metal, config)

    return x_t


def add_cbct_noise(x, config):
    image = recon_image(x)
    thresh = config['T3']
    metal = np.zeros_like(image)

    metal[image > thresh] = 1

    image_hu = hu2mu(image, config['mu_water'], config['mu_air'])

    x_t = ct_metal_artifact_simulation(image_hu, metal, config)

    return x_t


def main():
    parser = argparse.ArgumentParser('ddgan parameters')
    from util.ma_simul.ma_config import set_config_for_ct_artifact_simulation, set_config_for_cbct_artifact_simulation

    parser.add_argument('--ct_mean', type=float, default=0.0689)
    parser.add_argument('--ct_std', type=float, default=0.1156)
    parser.add_argument('--cbct_mean', type=float, default=0.0354)
    parser.add_argument('--cbct_std', type=float, default=0.0931)

    args = parser.parse_args()

    # Load images
    f = h5py.File(r"E:\workspace\research\SR\data\h5_cbct\validation_pCT.h5", "r")
    data_x = np.array(f["data"], np.float32)
    x = data_x[1][20]
    image = x

    pixel_size = 0.1  # [cm]
    config = set_config_for_ct_artifact_simulation(pixel_size)
    x_t = add_ct_noise(image, config)

    plt.imshow(x_t, cmap='gray')
    plt.title('metal artifacts sinogram')
    plt.show()


if __name__ == "__main__":
    main()
