import argparse
import numpy as np
import tensorflow as tf
from functools import partial

from utils_multistep_lr import *
from generator import *
from queues import *
from BWNet import BWNet

parser = argparse.ArgumentParser(description='tensorflow implementation of BWNet')

parser.add_argument('train_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'training cover images')
parser.add_argument('train_stego_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'training stego images or beta maps')
parser.add_argument('valid_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'validation cover images')
parser.add_argument('valid_stego_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'validation stego images or beta maps')
parser.add_argument('--is-testing',type=bool,default=False,help='determining whether doing test')
parser.add_argument('--testing-dir-cover',type=str,default='',help='testing dir')
parser.add_argument('--testing-dir-stego',type=str,default='',help='testing dir')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=4e-1, metavar='LR',
                    help='learning rate (default: 4e-1)')
parser.add_argument('--use-batch-norm', action='store_true', default=False,
                    help='use batch normalization after each activation,' +
                    ' also disable pair constraint (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0,
                    help='index of gpu used (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait ' +
                    'before logging training status')
parser.add_argument('--log-path', type=str, default='logs/',
                    metavar='PATH', help='path to generated log file')
parser.add_argument('--load-path', type=str, default='logs/',
                    metavar='PATH', help='path to load the model')
args = parser.parse_args()

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.no_cuda else str(args.gpu)

tf.set_random_seed(args.seed)
train_ds_size = len(glob(args.train_cover_dir + '/*')) * 2

train_gen = partial(gen_flip_and_rot, args.train_cover_dir, \
                        args.train_stego_dir)

valid_ds_size = len(glob(args.valid_cover_dir + '/*')) * 2
valid_gen = partial(gen_valid, args.valid_cover_dir, \
                    args.valid_stego_dir)
                    

if valid_ds_size % 32 != 0:
    raise ValueError("change batch size for validation")
    
optimizer = tf.train.AdadeltaOptimizer(args.lr)

if args.is_testing:
  test_ds_size = len(glob(args.test_dir_cover+'/*'))*2
  test_gen = partial(gen_test,args.test_dir_cover,args.test_dir_stego)
  test_dataset(BWNet,test_gen,args.test_batch_size,test_ds_size,args.load_path)
else:
  train(BWNet, train_gen, valid_gen, args.batch_size, \
      args.test_batch_size, valid_ds_size, \
      AdamaxOptimizer, [50000],[0.001,0.0001], args.log_interval,  args.log_interval,\
      500000, 1000, args.log_path,1)
