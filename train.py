import os
import sys
import time
import h5py
import json

import wandb
import pickle
import logging
import argparse
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

from shutil import copyfile
from collections import OrderedDict
import gc
from argparse import Namespace

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds

from mindspore import Model
from mindspore import ms_function
from mindspore import ParameterTuple
from mindspore import Tensor, set_seed, Parameter
from mindspore import DatasetHelper, connect_network_with_dataset
from mindspore.ops import ExpandDims
from mindspore.ops.composite import GradOperation
from mindspore.communication import get_rank, get_group_size
from mindspore.train.callback import ReduceLROnPlateau
from mindspore.communication import init, get_rank, get_group_size

from mindspore.experimental import optim


from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.darcy_loss import LpLoss, channel_wise_LpLoss
from utils.data_loader_multifiles import MyDataset
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from mindearth.cell import AFNONet
from mindearth.cell import GraphCastNet
from mindearth.utils import load_yaml_config


class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 

    def __init__(self, params):

        self.params = params



        if params.log_to_wandb:
            wandb.init(config=params, 
                       project=params.project,
                       )

        logging.info('begin data loader init' )

        logging.info(f'Train_data_path: {params.train_data_path}')
        my_dataset_trn = MyDataset(params, params.train_data_path, train=True)
        max_step_trn = my_dataset_trn.__len__()
        dataset_trn = ds.GeneratorDataset(my_dataset_trn, column_names=["inp","tar"], shuffle = True, num_shards=rank_size, shard_id=rank_id)
        self.dataset_trn = dataset_trn.batch(batch_size=args.batch_size, drop_remainder=False, num_parallel_workers=4)


        logging.info(f'Eval_data_path: {params.valid_data_path}')
        my_dataset_val = MyDataset(params, params.valid_data_path, train=True)
        max_step_val = my_dataset_val.__len__()
        dataset_val = ds.GeneratorDataset(my_dataset_val,column_names=["inp","tar"], shuffle = True, num_shards=rank_size, shard_id=rank_id)
        self.dataset_val = dataset_val.batch(batch_size=args.batch_size, drop_remainder=False, num_parallel_workers=4)
        logging.info('finish data loader init' )




        if params.loss_region_weighted:
            self.loss_obj = channel_wise_LpLoss(scale = params.loss_scale)
        elif params.loss_channel_wise:
            self.loss_obj = channel_wise_LpLoss(scale = params.loss_scale)
        else:
            self.loss_obj = LpLoss()

        logging.info('loss loaded')

        if params.nettype == 'afno':
            model = AFNONet(image_size=(720, 1440),
                            in_channels=67,
                            out_channels=61,
                            patch_size=8,
                            encoder_depths=16,
                            encoder_embed_dim=512,
                            mlp_ratio=4,
                            dropout_rate=1.)
        elif params.nettype == 'Masked_AE_Ocean':
            from networks.Masked_AE_Ocean import Masked_Ocean as model
            model = model(params, img_size=(720, 1440))
        else:
            raise Exception("not implemented")
        self.model = model

        logging.info('model loaded')

        self.iters = 0
        self.startEpoch = 0


        self.epoch = self.startEpoch


        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    factor=0.2, 
                    patience=5, 
                    mode='min'
            )
        elif params.scheduler == 'CosineAnnealingLR': 
            self.scheduler = nn.cosine_decay_lr(
                    min_lr=1e-4,
                    max_lr=1e-3,
                    total_step=params.max_epochs,
                    step_per_epoch=50,
                    decay_epoch=self.startEpoch+1
            )
        else:
            self.scheduler = None


        def switch_off_grad(self, model):
            for param in model.parameters():
                param.requires_grad = False



        initial_lr = 0.001
        min_lr = 0.0001
        decay_steps = 900

        lr_scheduler = nn.CosineDecayLR(min_lr, initial_lr, decay_steps)


        milestone = [50, 100, 500]
        learning_rates = [0.001, 0.0005, 0.0001]
        lr_dynamic = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)


        if params.optimizer_type == 'FusedAdam':
            self.optimizer = nn.AdamWeightDecay(self.model.trainable_params(), learning_rate = 0.0001, weight_decay=params.weight_decay)
        else:
            self.optimizer = optim.AdamW(self.model.trainable_params(), lr= lr_scheduler,weight_decay=0.0)
        logging.info('optimizer_type loaded')

    def train(self):
        if self.params.log_to_screen:
            logging.info("==================Starting Training Loop==================")

        best_valid_loss = 1.e6
        eval_patience = 0

        for epoch in range(self.startEpoch, self.params.max_epochs):

            start = time.time()
            tr_time, data_time, step_time, train_logs = self.train_one_epoch() 
            valid_time, valid_logs = self.validate_one_epoch()


            if epoch == self.params.max_epochs - 1 and self.params.prediction_type == 'direct':
                valid_weighted_rmse = self.validate_final()


            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})



            
            logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            logging.info('train data time={}, train per epoch time={}, train per step time={}, valid time={}'.format(data_time, tr_time, step_time, valid_time))
            logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['train_loss'], valid_logs['valid_loss']))


            def save_checkpoint_with_retry(model, path, retries=3, delay=5):
                for attempt in range(retries):
                    try:
                        ms.save_checkpoint(model, path)
                        return True
                    except Exception as e:
                        logging.critical(f"Failed to save the checkpoint file {path} on attempt {attempt + 1}. Error: {e}")
                        if attempt < retries - 1:
                            time.sleep(delay)
                        else:
                            raise e
                return False

            if self.params.save_checkpoint:
                save_checkpoint_with_retry(self.model, self.params.checkpoint_path)
                if valid_logs['valid_loss'] <= best_valid_loss:
                    save_checkpoint_with_retry(self.model, self.params.best_checkpoint_path)
                    

            

            if valid_logs['valid_loss'] <= best_valid_loss:
                eval_patience = 0
                best_valid_loss = valid_logs['valid_loss']
            else:
                eval_patience += 1





    def land_mask_func(self, x, y, land_mask_path):

        
        with h5py.File(land_mask_path, 'r') as _f: 
            mask_data = np.array(_f['fields'], dtype=np.float32)
            mask_data = Tensor.from_numpy(mask_data)
            out_channels_indices = self.params.out_channels.tolist() 
            mask_data = mask_data[0,out_channels_indices]
        x = ops.masked_fill(input_x=x, mask=~mask_data, value=0)
        y = ops.masked_fill(input_x=y, mask=~mask_data, value=0)
        return x, y 



    def train_one_epoch(self):
        self.model.set_train()

        def forward_fn(inp, tar):
            outputs = self.model(inp)
            target_shape = (1, 61, 720, 1440)
            outputs = outputs.reshape(target_shape)
            
            if self.params.land_mask:
                outputs, tar = self.land_mask_func(outputs, tar, self.params.land_mask_path)
            loss = self.loss_obj(outputs, tar)
            return loss

        grad_fn = ms.value_and_grad(forward_fn, None,self.optimizer.parameters)

        def train_step(inp, tar):
            loss, grads = grad_fn(inp, tar)
            self.optimizer(grads)
            return loss


        self.epoch += 1
        tr_time = 0
        data_time = 0
        

        steps_in_one_epoch = 0
        train_loss = 0

        for i, data in enumerate(self.dataset_trn.create_tuple_iterator()):
            self.iters += 1
            steps_in_one_epoch += 1 
            print("steps_in_one_epoch:",steps_in_one_epoch)

            data_start = time.time()
            (inp, tar) = data
            if self.params.orography and self.params.multi_steps_finetune > 1:
                orog = ops.unsqueeze(inp[:,-1], dim=1)
            data_time += time.time() - data_start
            tr_start = time.time()
            loss = train_step(inp, tar)
            train_loss = loss.asnumpy()
            tr_time += time.time() - tr_start

        logs = {'train_loss': train_loss}
        print('train loss',train_loss)

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)
        step_time = tr_time / steps_in_one_epoch

        return tr_time, data_time, step_time, logs

    def validate_one_epoch(self):


        valid_buff  = ops.zeros((3+self.params.N_out_channels))
        valid_loss  = valid_buff[0].view(-1) 
        valid_l1    = valid_buff[1].view(-1) 
        valid_steps = valid_buff[-1].view(-1) 

        valid_start = time.time()


        for i, data in enumerate(self.dataset_val.create_tuple_iterator()):

            
            inp, tar = map(lambda x: x, data)
            gen = self.model(inp)
            target_shape = (1, 61, 720, 1440)
            gen = gen.reshape(target_shape)


            if self.params.multi_steps_finetune > 1:
                tar = tar[:, 0, self.params.out_channels] 

            if self.params.land_mask:
                gen, tar = self.land_mask_func(gen, tar, self.params.land_mask_path)
            valid_loss_ = self.loss_obj(gen, tar)

            valid_steps += 1.

        valid_time = time.time() - valid_start
        
        logs = {'valid_loss': valid_loss_,}

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return valid_time, logs

    
    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        ms.save_checkpoint(model, checkpoint_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/Model_202312.yaml', type=str)  
    parser.add_argument("--multi_steps_finetune", default=1, type=int)  
    parser.add_argument("--finetune_max_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--wandb_group", default='025_daily', type=str)
    parser.add_argument("--config", default='Masked_AE_Ocean', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--gpu_id", default='0', type=str)
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config, True)
    params['epsilon_factor'] = args.epsilon_factor
    params['multi_steps_finetune'] = args.multi_steps_finetune
    params['finetune_max_epochs']  = args.finetune_max_epochs


    ms.set_context(mode = ms.context.PYNATIVE_MODE,pynative_synchronize = True,device_target='GPU')
    ms.set_seed(1)
    init("nccl")
    rank_num = get_group_size()
    rank_id = get_rank()
    rank_size = rank_num
    print("rank_id is {}, device_num is {}".format(rank_id, rank_num))
    ms.set_auto_parallel_context(dataset_strategy="data_parallel", device_num=rank_num, parallel_mode=ms.context.ParallelMode.DATA_PARALLEL, gradients_mean=True)

    params['batch_size'] = args.batch_size  
    params['enable_amp'] = args.enable_amp 


    if params['multi_steps_finetune'] > 1:
        pretrained_expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
        params['pretrained_ckpt_path'] = os.path.join(pretrained_expDir, 'training_checkpoints/best_ckpt.ckpt')

        multi_steps = params['multi_steps_finetune']

        expDir = os.path.join(pretrained_expDir, f'{multi_steps}_steps_finetune')
        if world_rank == 0:
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)

        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.ckpt') 
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.ckpt')

        params['resuming'] = True
    else:
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
        os.makedirs(expDir, exist_ok =True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok =True)
        copyfile(os.path.abspath(args.yaml_config), os.path.join(expDir, 'config.yaml'))

        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.ckpt') 
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.ckpt')


        args.resuming = True if os.path.isfile(params.checkpoint_path) else False
        params['resuming'] = args.resuming


    params['entity'] = "ocean_ai_model"  
    params['project'] = "ai-goms_mindspore"    
    params['group'] = args.wandb_group  
    params['name'] = args.config + '_' + str(args.run_num)  


    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'train.log'))
    logging_utils.log_versions()
    params.log()

    params['log_to_wandb'] = (rank_id == 0) and params['log_to_wandb']
    params['log_to_screen'] = (rank_id == 0) and params['log_to_screen']

    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    params['N_out_channels'] = len(params['out_channels'])
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) + 1
    else:
        params['N_in_channels'] = len(params['in_channels']) 

    if rank_id == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params)
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)
