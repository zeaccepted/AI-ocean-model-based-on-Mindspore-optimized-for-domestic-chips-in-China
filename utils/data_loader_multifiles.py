

import logging
import glob

import random
import numpy as np

import h5py
import math

from utils.img_utils import reshape_fields, reshape_finetune_fields
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.nn import RMSELoss
from mindspore import dtype as mstype
from mindspore.ops import operations as P
import mindspore.numpy as mnp



class MyDataset:
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.orography = params.orography
        self.normalize = params.normalize
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.atmos_channels = np.array(params.atmos_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        self._get_files_stats()
        self.add_noise = params.add_noise if train else False
        self.fusion_3d_2d = params.fusion_3d_2d


    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f: 
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0] - self.params.multi_steps_finetune 

            self.img_shape_x = _f['fields'].shape[2] - 1
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location,
                                                                                                       self.n_samples_total,
                                                                                                       self.img_shape_x,
                                                                                                       self.img_shape_y,
                                                                                                       self.n_in_channels))
        logging.info("Delta t: {} days".format(1 * self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(
            1 * self.dt * self.n_history, 1 * self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields'] 

        if self.orography and self.params.normalization == 'zscore': 
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, 'r')
        if self.orography and self.params.normalization == 'maxmin': 
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, 'r')
        self.orography_field = _orog_file['orog']

    def __len__(self):
        return self.n_samples_total


    def vis_label_distribution(self):
        print('\nVisibility distribution: ')
        print(f'min: {np.min(self.label)}, max: {np.max(self.label)}')

    def __getitem__(self, index):
        year_idx  = int(index / self.n_samples_per_year)
        local_idx = int(index % self.n_samples_per_year)

        if self.files[year_idx] is None:
            self._open_file(year_idx)

        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field 
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
        else:
            orog = None
        
        if self.params.multi_steps_finetune == 1:
            inp = reshape_fields( 
                    self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 
                    'inp', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )
            tar = reshape_fields(
                    self.files[year_idx][local_idx+step, self.out_channels], 
                    'tar', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog 
                )
        else:
            inp = reshape_fields( 
                    self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 
                    'inp', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )
            tar = reshape_fields( 
                    self.files[year_idx][local_idx+step:local_idx+step+self.params.multi_steps_finetune, self.in_channels], 
                    'tar', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog 
                )

        return inp, tar 

    def __len__(self):
        return self.n_samples_total



class GetDataset_Wave:
    def __init__(self, params, backbone_data_location, finetune_data_location, train):

        self.train = train
        self.params = params
        self.backbone_data_location = backbone_data_location
        self.finetune_data_location = finetune_data_location

        self.orography = params.orography
        self.normalize = params.normalize

        self.dt = params.dt              
        self.n_history = params.n_history

        # backbone data channels
        self.in_channels    = np.array(params.in_channels)
        self.out_channels   = np.array(params.out_channels)
        self.n_in_channels  = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        # finetune data channels
        self.finetune_in_channels    = np.array(params.finetune_in_channels)
        self.finetune_out_channels   = np.array(params.finetune_out_channels)
        self.finetune_force_channels  = np.array(params.finetune_force_channels)
        self.finetune_n_in_channels  = len(self.finetune_in_channels)
        self.finetune_n_out_channels = len(self.finetune_out_channels)

        self._get_backbone_files_stats()
        self._get_finetune_task_files_stats()

        self.add_noise = params.add_noise if train else False

    def _get_backbone_files_stats(self):
        self.files_paths = glob.glob(self.backbone_data_location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f: 
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0] - self.dt 


            self.img_shape_x = _f['fields'].shape[2] - 1
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info("Found backbone data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.backbone_data_location,
            self.n_samples_total,
            self.img_shape_x,
            self.img_shape_y,
            self.n_in_channels))
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(self.dt*self.n_history, self.dt))

    def _get_finetune_task_files_stats(self):

        self.finetune_files_paths = glob.glob(self.finetune_data_location + "/*.h5")
        self.finetune_files_paths.sort()
        self.finetune_n_years = len(self.finetune_files_paths)

        with h5py.File(self.finetune_files_paths[0], 'r') as _f: 
            logging.info("Getting finetune task file stats from {}".format(self.finetune_files_paths[0]))
            self.finetune_n_samples_per_year = _f['fields'].shape[0] - 1 


            self.finetune_img_shape_x = _f['fields'].shape[2] - 1
            self.finetune_img_shape_y = _f['fields'].shape[3]

        self.finetune_n_samples_total = self.finetune_n_years * self.finetune_n_samples_per_year
        self.finetune_files = [None for _ in range(self.finetune_n_years)]

        logging.info("Found finetune task data at path {}. Number of examples: {}. Image Shape: {} x {}".format(
            self.finetune_data_location,
            self.finetune_n_samples_total,
            self.finetune_img_shape_x,
            self.finetune_img_shape_y))

        logging.info("Number of finetune samples per year: {}".format(self.finetune_n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(self.dt*self.n_history, self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields'] 

        if self.orography and self.params.normalization == 'zscore': 
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, 'r')
        if self.orography and self.params.normalization == 'maxmin': 
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, 'r')
        self.orography_field = _orog_file['orog']

    def _open_finetune_file(self, year_idx):
        _file = h5py.File(self.finetune_files_paths[year_idx], 'r')
        self.finetune_files[year_idx] = _file['fields'] 

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx  = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year
        if self.files[year_idx] is None:
            self._open_file(year_idx)
            self._open_finetune_file(year_idx)


        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field 
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
        else:
            orog = None

        inp = reshape_fields( self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.params, self.train, self.normalize, orog, self.add_noise )
        tar = reshape_fields( self.files[year_idx][local_idx+step, self.out_channels], 'tar', self.params, self.train, self.normalize, orog )
        
        inp_finetune = reshape_finetune_fields( self.finetune_files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.finetune_out_channels], 'tar', self.params, self.normalize )
        inp_wind_finetune = reshape_finetune_fields( self.finetune_files[year_idx][local_idx+step, self.finetune_force_channels], 'force', self.params, self.normalize )
        
        inp_finetune = np.expand_dims(inp_finetune,0)

        print('inp_finetune',inp_finetune.shape)
        print('inp_wind_finetune',inp_wind_finetune.shape)

        tar_finetune = reshape_finetune_fields( self.finetune_files[year_idx][local_idx+step, self.finetune_out_channels], 'tar', self.params, self.normalize )

        print(f'inp: {inp.shape}, tar: {tar.shape}')
        print(f'inp_finetune: {inp_finetune.shape}, tar_finetune: {tar_finetune.shape}')

        return inp, tar, inp_finetune, tar_finetune , inp_wind_finetune


class GetDataset_Kuroshio_Downscaling:
    def __init__(self, params, backbone_data_location, finetune_data_location, train):

        self.train = train
        self.params = params
        self.backbone_data_location = backbone_data_location
        self.finetune_data_location = finetune_data_location

        self.orography = params.orography
        self.normalize = params.normalize

        self.dt = params.dt              
        self.n_history = params.n_history 

        # backbone data channels
        self.in_channels    = np.array(params.in_channels)
        self.out_channels   = np.array(params.out_channels)
        self.n_in_channels  = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        # finetune data channels
        self.finetune_in_channels    = np.array(params.finetune_in_channels)
        self.finetune_out_channels   = np.array(params.finetune_out_channels)
        self.finetune_n_in_channels  = len(self.finetune_in_channels)
        self.finetune_n_out_channels = len(self.finetune_out_channels)

        self._get_backbone_files_stats()
        self._get_finetune_task_files_stats()

        self.add_noise = params.add_noise if train else False

    def _get_backbone_files_stats(self):
        self.files_paths = glob.glob(self.backbone_data_location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f: 
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0] - 1 

            # original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2] 
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info("Found backbone data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.backbone_data_location,
            self.n_samples_total,
            self.img_shape_x,
            self.img_shape_y,
            self.n_in_channels))
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(self.dt*self.n_history, self.dt))

    def _get_finetune_task_files_stats(self):

        self.finetune_files_paths = glob.glob(self.finetune_data_location + "/*.h5")
        self.finetune_files_paths.sort()
        self.finetune_n_years = len(self.finetune_files_paths)

        with h5py.File(self.finetune_files_paths[0], 'r') as _f: 
            logging.info("Getting finetune task file stats from {}".format(self.finetune_files_paths[0]))
            self.finetune_n_samples_per_year = _f['fields_0p08'].shape[0] - 1 

            # original image shape (before padding)
            self.finetune_0p25_img_shape_x = _f['fields_0p25'].shape[2]
            self.finetune_0p25_img_shape_y = _f['fields_0p25'].shape[3]

            self.finetune_0p08_img_shape_x = _f['fields_0p08'].shape[2]
            self.finetune_0p08_img_shape_y = _f['fields_0p08'].shape[3]

        self.finetune_n_samples_total = self.finetune_n_years * self.finetune_n_samples_per_year
        self.finetune_files_0p08 = [None for _ in range(self.finetune_n_years)]
        self.finetune_files_0p25 = [None for _ in range(self.finetune_n_years)]

        logging.info("Found finetune task data at path {}. Number of examples: {}. 0p25 Image Shape: {} x {}. 0p08 Image Shape: {} x {}".format(
            self.finetune_data_location,
            self.finetune_n_samples_total,
            self.finetune_0p25_img_shape_x,
            self.finetune_0p25_img_shape_y,
            self.finetune_0p08_img_shape_x,
            self.finetune_0p08_img_shape_y))

        logging.info("Number of finetune samples per year: {}".format(self.finetune_n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(self.dt*self.n_history, self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields'] 

        if self.orography and self.params.normalization == 'zscore': 
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, 'r')
        if self.orography and self.params.normalization == 'maxmin': 
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, 'r')
        self.orography_field = _orog_file['orog']

    def _open_finetune_file(self, year_idx):
        _file = h5py.File(self.finetune_files_paths[year_idx], 'r')
        self.finetune_files_0p08[year_idx] = _file['fields_0p08'] 
        self.finetune_files_0p25[year_idx] = _file['fields_0p25'] 

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx  = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)
            self._open_finetune_file(year_idx)

        # If there are not enough historical time steps available in the features, shift to future time steps.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # If the sample is the final one for the year, predict the current time step. Otherwise, predict the next time step.
        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field 
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
        else:
            orog = None

        inp = reshape_fields( self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.params, self.train, self.normalize, orog, self.add_noise )
        tar = reshape_fields( self.files[year_idx][local_idx+step, self.out_channels], 'tar', self.params, self.train, self.normalize, orog )
        inp_finetune = reshape_finetune_fields( self.finetune_files_0p25[year_idx][local_idx+step, self.finetune_in_channels], 'inp', self.params, self.normalize )
        tar_finetune = reshape_finetune_fields( self.finetune_files_0p08[year_idx][local_idx+step, self.finetune_out_channels], 'tar', self.params, self.normalize )

        return inp, tar, inp_finetune, tar_finetune 

class GetDataset_Biochemical:
    def __init__(self, params, backbone_data_location, finetune_data_location, train):

        self.train = train
        self.params = params
        self.backbone_data_location = backbone_data_location
        self.finetune_data_location = finetune_data_location

        self.orography = params.orography
        self.normalize = params.normalize

        self.dt = params.dt              
        self.n_history = params.n_history # 0

        # backbone data channels
        self.in_channels    = np.array(params.in_channels)
        self.out_channels   = np.array(params.out_channels)
        self.n_in_channels  = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        # finetune data channels
        self.finetune_in_channels    = np.array(params.finetune_in_channels)
        self.finetune_out_channels   = np.array(params.finetune_out_channels)
        self.finetune_n_in_channels  = len(self.finetune_in_channels)
        self.finetune_n_out_channels = len(self.finetune_out_channels)

        self._get_backbone_files_stats()
        self._get_finetune_task_files_stats()

        self.add_noise = params.add_noise if train else False

    def _get_backbone_files_stats(self):
        self.files_paths = glob.glob(self.backbone_data_location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f: 
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0] - 1 

            # original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info("Found backbone data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.backbone_data_location,
            self.n_samples_total,
            self.img_shape_x,
            self.img_shape_y,
            self.n_in_channels))
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(self.dt*self.n_history, self.dt))

    def _get_finetune_task_files_stats(self):

        self.finetune_files_paths = glob.glob(self.finetune_data_location + "/*.h5")
        self.finetune_files_paths.sort()
        self.finetune_n_years = len(self.finetune_files_paths)

        with h5py.File(self.finetune_files_paths[0], 'r') as _f: 
            logging.info("Getting finetune task file stats from {}".format(self.finetune_files_paths[0]))
            self.finetune_n_samples_per_year = _f['fields'].shape[0] - 1 

            # original image shape (before padding)
            self.finetune_img_shape_x = _f['fields'].shape[2]
            self.finetune_img_shape_y = _f['fields'].shape[3]

        self.finetune_n_samples_total = self.finetune_n_years * self.finetune_n_samples_per_year
        self.finetune_files = [None for _ in range(self.finetune_n_years)]

        logging.info("Found finetune task data at path {}. Number of examples: {}. Image Shape: {} x {}".format(
            self.finetune_data_location,
            self.finetune_n_samples_total,
            self.finetune_img_shape_x,
            self.finetune_img_shape_y))

        logging.info("Number of finetune samples per year: {}".format(self.finetune_n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(self.dt*self.n_history, self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields'] 

        if self.orography and self.params.normalization == 'zscore': 
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, 'r')
        if self.orography and self.params.normalization == 'maxmin': 
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, 'r')
        self.orography_field = _orog_file['orog']

    def _open_finetune_file(self, year_idx):
        # print('year_idx',year_idx)
        _file = h5py.File(self.finetune_files_paths[year_idx], 'r')
        self.finetune_files[year_idx] = _file['fields'] 

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):

        year_idx  = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)
            self._open_finetune_file(year_idx)

        # If there are not enough historical time steps available in the features, shift to future time steps.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # If the sample is the final one for the year, predict the current time step. Otherwise, predict the next time step.
        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field 
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
        else:
            orog = None

        inp = reshape_fields( self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.params, self.train, self.normalize, orog, self.add_noise )
        tar = reshape_fields( self.files[year_idx][local_idx+step, self.out_channels], 'tar', self.params, self.train, self.normalize, orog )
        inp_finetune = reshape_finetune_fields( self.finetune_files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.finetune_in_channels], 'inp', self.params, self.normalize )
        tar_finetune = reshape_finetune_fields( self.finetune_files[year_idx][local_idx+step, self.finetune_out_channels], 'tar', self.params, self.normalize )

        return inp, tar, inp_finetune, tar_finetune 
