import numpy as np
import pandas as pd
from ma_simul.build_geometry import initialization, imaging_geo

n_proj = 256
param = initialization(n_proj)
# csv_path = './xray_characteristic_data.csv'
csv_path = '/shenlab/lab_stor/yunkuipa/codes_cvpr_simul/util/ma_simul/xray_characteristic_data.csv'


def set_config_for_ct_artifact_simulation(pixel_size):
    config = {}
    # basic
    metals = ['Titanium', 'Iron']
    config['data'] = pd.read_csv(csv_path)
    config['E0'] = 40  # equivalent monochromatic energy [keV]
    config['metal_name'] = metals[1]  # used metal
    config['mu_water'] = config['data'].loc[config['E0'], 'Water']  # linear coefficient of water with E0
    config['mu_air'] = 0  # linear coefficient of air with E0
    config['T1'] = 150  # soft tissue threshold for threshold-based weighting
    config['T2'] = 1300#1500  # bone threshold for threshold-based weighting
    config['T3'] = 1000#1000  # bone threshold for threshold-based weighting
    # sampled energy for polychromatic projection
    config['energy_composition'] = np.arange(0, 120, dtype=np.int32)
    config['polynomial_order_for_correction'] = 3  # degree of polynomial fit used in water correction
    config['m_water'] = config['data']['Water']
    config['m_bone'] = config['data']['Bone']
    config['m_metal'] = config['data'][config['metal_name']]
    config['m_intensity'] = config['data']['Intensity']
    config['correction_coeff'] = np.asarray([[-1.04811676e-02], [9.82882828e-02], [9.33561802e-01], [4.56019654e-04]])

    # ============= fiexible =============
    config['metal_density'] = 1.5
    config['bone_level'] = 1  # used metal
    config['water_level'] = 1  # density of the metal
    r = (config['metal_density'] ** 2 - 1)
    config['metal_level'] = 2. / (1 + np.exp(-r)) - 1
    # (1, 201, step=2.)
    config['noise_scale'] = 1e2  # variance of poisson noise
    config['percent_stripe'] = 0.1
    config['min_v'] = 0.8
    config['max_v'] = 1.8
    config['stripe_rate'] = 0.2
    config['blur_sigma'] = 2.0

    # geometric parameters
    config['pixel_size'] = pixel_size  # the real size of each pixel [cm]

    fp, bp = imaging_geo(param)

    config['fanbeam'] = fp
    config['ifanbeam'] = bp

    return config


def set_config_for_cbct_artifact_simulation(pixel_size):
    config = {}
    # basic
    metals = ['Titanium', 'Iron']
    config['data'] = pd.read_csv(csv_path)
    config['E0'] = 40  # equivalent monochromatic energy [keV]
    config['metal_name'] = metals[1]  # used metal
    config['mu_water'] = config['data'].loc[config['E0'], 'Water']  # linear coefficient of water with E0
    config['mu_air'] = 0  # linear coefficient of air with E0
    config['T1'] = 150  # soft tissue threshold for threshold-based weighting
    config['T2'] = 1500  # bone threshold for threshold-based weighting
    config['T3'] = 2400  # bone threshold for threshold-based weighting
    # sampled energy for polychromatic projection
    config['energy_composition'] = np.arange(0, 120, dtype=np.int32)
    config['polynomial_order_for_correction'] = 3  # degree of polynomial fit used in water correction
    config['m_water'] = config['data']['Water']
    config['m_bone'] = config['data']['Bone']
    config['m_metal'] = config['data'][config['metal_name']]
    config['m_intensity'] = config['data']['Intensity']
    config['correction_coeff'] = np.asarray([[-1.04811676e-02], [9.82882828e-02], [9.33561802e-01], [4.56019654e-04]])

    # ============= fiexible =============
    config['metal_density'] = 1.0  # density of the metal
    config['bone_level'] = 1.0  # used metal
    config['water_level'] = 1.0  # density of the metal
    # (1, 201, step=2.)
    config['noise_scale'] = 1e1  # variance of poisson noise
    r = (config['metal_density'] ** 2 - 1)
    config['metal_level'] = 2. / (1 + np.exp(-r)) - 1
    # Scale factor for rescaling the frequency axis, specified as a positive number in the range (0, 1]
    config['freqscale'] = 1
    config['blur_sigma'] = None
    # config['blur'] = T.GaussianBlur(5, sigma=5.0)

    # geometric parameters
    config['pixel_size'] = pixel_size  # the real size of each pixel [cm]

    fp, bp = imaging_geo(param)

    config['fanbeam'] = fp
    config['ifanbeam'] = bp

    return config
