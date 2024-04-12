import argparse
import logging
import os
import time

import torch.optim as optim
import torch.utils.data
from pytorch.pytorch_utils import move_data_to_device, count_parameters, do_mixup
from utils import config
from utils.data_generator import AudioSetDatasetCsv, collate_fn, CsvTrainSampler
from utils.utilities import (create_folder, get_filename, create_logging, Mixup, 
    )
from pytorch.models import Cnn10

from pytorch.losses import get_loss_func


def train(args):
    """Train AudioSet tagging model. 

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    train_csv_path = args.train_csv_path
    classes_num = args.classes_num

    num_workers = 8
    loss_func = get_loss_func(loss_type)

    # Paths
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    # Model
    Model = Cnn10
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
     
    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))
    
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = AudioSetDatasetCsv(classes_num, sample_rate=sample_rate)

    train_sampler = CsvTrainSampler(train_csv_path, batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Resume training
    iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    time1 = time.time()
    
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        # Save model
        if iteration % 100 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                }

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # Mixup lambda
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], 
                batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            # print(batch_output_dict)

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""
            # print(batch_target_dict)

        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)

        # Backward
        loss.backward()
        print(loss)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if iteration % 10 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'\
                .format(iteration, time.time() - time1))
            time1 = time.time()
        
        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Use this program to train')

    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000) 
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce'])
    parser.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser.add_argument('--augmentation', type=str, default='none', choices=['none', 'mixup'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--resume_iteration', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=1000000)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--train_csv_path')
    parser.add_argument('--classes_num', default=config.classes_num, type=int)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    train(args)
