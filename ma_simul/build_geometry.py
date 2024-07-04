import numpy as np


class initialization:
    def __init__(self, n_proj):
        self.param = {}
        self.reso = 512 / 256 * 0.03

        # image
        self.param['nx_h'] = 256
        self.param['ny_h'] = 256
        self.param['sx'] = self.param['nx_h'] * self.reso
        self.param['sy'] = self.param['ny_h'] * self.reso

        ## view
        self.param['startangle'] = 0
        self.param['endangle'] = 2 * np.pi

        self.param['nProj'] = 256

        ## detector
        self.param['su'] = 2 * np.sqrt(self.param['sx'] ** 2 + self.param['sy'] ** 2)
        self.param['nu_h'] = 256
        self.param['dde'] = 1075 * self.reso
        self.param['dso'] = 1075 * self.reso

        self.param['u_water'] = 0.192


def imaging_geo(param):
    import odl
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')
    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])

    detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
                                                 param.param['nu_h'])

    geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
                                          src_radius=param.param['dso'],
                                          det_radius=param.param['dde'])

    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cpu')
    FBPOper_hh = odl.tomo.fbp_op(ray_trafo_hh, filter_type='Hann', frequency_scaling=1.0)

    return ray_trafo_hh, FBPOper_hh


def imaging_geo_cuda(param):
    import odl
    from odl.contrib import torch as odl_torch
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')
    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])

    detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
                                                 param.param['nu_h'])

    geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
                                          src_radius=param.param['dso'],
                                          det_radius=param.param['dde'])

    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')

    return odl_torch.OperatorModule(ray_trafo_hh)



def imaging_geo_parallel(param):
    space = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')
    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])
    detector_partition = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
                                               param.param['nu_h'])
    detector = odl.tomo.geometry.Flat1dDetector(detector_partition, axis=[256, 0], check_bounds=True)
    init_pos = np.zeros(2)
    geometry = odl.tomo.ParallelBeamGeometry(2, angle_partition, detector, init_pos)
    print(geometry.angles.size)

    ray_trafo_hh = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
    FBPOper_hh = odl.tomo.fbp_op(ray_trafo_hh, filter_type='Ram-Lak')

    return ray_trafo_hh, FBPOper_hh
