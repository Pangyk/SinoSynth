import numpy as np
from scipy import ndimage

from numpy.random import default_rng
import cv2
import skimage.exposure


def norm(x, mean, std):
    x = np.clip(x, -500, 1000)
    x = (x + 500) / 1500
    return x


def hu2mu(hu, mu_water, mu_air):
    # convert from HU to linear attenuation coefficient (mu)
    mu = hu / 1000.0 * (mu_water - mu_air) + mu_water
    return mu


def mu2hu(mu, mu_water, mu_air):
    # convert from mu (linear attenuation coefficient) to HU (Hounsfield Unit)
    hu = 1000 * (mu - mu_water) / (mu_water - mu_air)
    return hu


def threshold_based_weighting(image, T1, T2):
    # apply weight function to the image based on given two thresholds
    w_bone = (image - T1) / (T2 - T1)
    w_bone = np.clip(w_bone, 0, 1)
    bone = w_bone * image

    w_water = (T2 - image) / (T2 - T1)
    w_water = np.clip(w_water, 0, 1)
    water = w_water * image

    return water, bone


def clip(a, minimum, maximum):
    # clip values in tensor 'a' between 'minimum' and 'maximum'
    clipped = a
    clipped[a > maximum] = maximum
    clipped[a < minimum] = minimum
    return clipped


def make_mask(x, p, min_v, max_v):
    m = np.ones_like(x)
    num = int(m.shape[-1] * p)
    stripe = np.random.choice(np.arange(0, m.shape[-1], dtype=np.int32), num, replace=False)
    value = np.random.uniform(min_v, max_v, stripe.shape)
    thickness = np.random.randint(-3, 10, num)
    for i, sp in enumerate(stripe):
        if thickness[i] < 1:
            thickness[i] = 1
        m[:, stripe[i]:stripe[i] + thickness[i]] = value[i]

    return m


def make_holes(img, sx, sy):
    height, width = img.shape
    # define random seed to change the pattern
    rng = default_rng()
    # create random noise image
    noise = rng.integers(0, 255, (height, width), np.uint8, True)
    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_DEFAULT)
    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 155, 235, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def block_artifact(x, seed, scale_bright, scale_dark):
    main_body = np.ones_like(x)
    T = hu2mu(-300, 0.286, 0)
    main_body[x < T] = 0
    mask1 = np.asarray(make_holes(x, seed[0], seed[1]), dtype=np.float64)
    mask2 = np.asarray(make_holes(x, seed[2], seed[3]), dtype=np.float64)
    mask1[mask1 > 0] = 1
    mask2[mask2 > 0] = 1

    m = mask1 - mask2
    m[m > 0] = 0
    mask2 = np.abs(m)

    mask = np.ones_like(x)
    mask1 *= np.random.uniform(low=0.0, high=1.0, size=x.shape)
    mask2 *= np.random.uniform(low=0.0, high=1.0, size=x.shape)
    if np.random.rand(1)[0] > 0.5:
        mask += mask1 * (scale_bright - 1)
    else:
        mask += mask1 * (scale_bright - 1) + mask2 * (scale_dark - 1)

    img = x * mask * main_body
    return img


def ct_metal_artifact_simulation(image, x_metal, config):
    # parse arguments
    data = config['data']
    energy_composition = config['energy_composition']
    E0 = config['E0']
    mu_air = config['mu_air']
    metal_name = config['metal_name']
    metal_density = config['metal_density']
    T1 = config['T1']
    T2 = config['T2']
    r = config['metal_level']
    correction_coeff = config['correction_coeff']
    pixel_size = config['pixel_size']

    m0_water = data['Water'][E0]
    m0_bone = data['Bone'][E0]
    m0_metal = data[metal_name][E0]
    mu_water0 = m0_water * 1.0
    mu_metal0 = ((m0_metal - m0_bone) * r + m0_bone) * metal_density

    # Threshold-based weighting
    T1 = hu2mu(T1, mu_water0, mu_air)
    T2 = hu2mu(T2, mu_water0, mu_air)
    x_water, x_bone = threshold_based_weighting(image, T1, T2)
    x_water *= config['water_level']
    x_bone *= config['bone_level']
    # d_water_full = config['fanbeam'](x_water) * pixel_size
    # d_bone_full = config['fanbeam'](x_bone) * pixel_size
    x_water[x_metal > 0] = 0
    x_bone[x_metal > 0] = 0
    x_metal_o = x_metal * mu_metal0
    lam = config['noise_scale']
    degree, blur_sigma, bright, dark = np.random.uniform((0.0, 0.0, 1.1, 0.4), (180.0, 2.0, 1.5, 0.9), 4)
    if config['blur_sigma'] is not None:
        mask = make_mask(x_water, config['percent_stripe'], config['min_v'], config['max_v'])
        x_water = ndimage.rotate(mask, degree, reshape=False) * x_water
        # x_water = np.clip(x_water, 0, 1.0)
        seed = np.random.randint(1, 15, 4)
        # remove block artifact
        # x_water = block_artifact(x_water, seed, bright, dark)
    else:
        blur_sigma = 0.0
    if blur_sigma >= 0.5:
        x_water = ndimage.gaussian_filter(x_water, sigma=blur_sigma)

    # Forward Projection
    d_water = config['fanbeam'](x_water) * pixel_size
    d_bone = config['fanbeam'](x_bone) * pixel_size
    d_metal = config['fanbeam'](x_metal_o) * pixel_size

    # Energy Composition
    m_water = config['m_water'][energy_composition]
    m_bone = config['m_bone'][energy_composition]
    m_metal = config['m_metal'][energy_composition]
    intensity = config['m_intensity'][energy_composition]

    d_water_tmp = np.einsum("ij,k->ijk", d_water, m_water / m0_water)
    d_bone_tmp = np.einsum("ij,k->ijk", d_bone, m_bone / m0_bone)
    d_metal_tmp = np.einsum("ij,k->ijk", d_metal, (m_metal / m0_metal) * r + (m_bone / m0_bone) * (1 - r))

    DRR = d_water_tmp + d_bone_tmp + d_metal_tmp

    y = np.einsum("ijk,k->ijk", (np.exp(-DRR)), intensity)
    total_intensity = np.sum(intensity)

    poly_y = np.sum(y, axis=2)
    if config['blur_sigma'] is not None:
        poly_y = lam * np.random.poisson(poly_y / lam)

    # Reconstruction
    x_ma = calibration(poly_y, total_intensity, correction_coeff, config)

    return x_ma


def calibration(p, total_intensity, correction_coeff, config):
    ps = -np.log(p / total_intensity)
    ps = polyval(correction_coeff, ps)
    sim = np.asarray(config['ifanbeam'](ps))
    sim[sim < 0] = 0
    sim = sim / config['pixel_size']

    s = mu2hu(sim, config['mu_water'], config['mu_air'])
    s = np.clip(s, -500, 1000)

    return s


def polyval(c, p):
    y = 0
    for pv in c:
        y = y * p + pv

    return y
