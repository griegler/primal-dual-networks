#!/usr/bin/env python2

import socket
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import time
import argparse
import os
import h5py
from glob import glob
import cv2
import random


#-------------------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------------------
def im_clamp(im):
  im_min = im.min()
  im_max = im.max()
  im -= im_min
  im /= (im_max - im_min)
  return im

def h5_write(h5_path, patches):
  print(h5_out_path)
  h5_file = h5py.File(h5_out_path, 'w')

  perm = None
  for key in patches:
    data = patches[key]
    if perm is None:
      perm = range(len(data))
      random.shuffle(perm)
    data = np.array(data)[perm]
    print(key, data.shape)
    chunks = (min(data.shape[0], 1000), data.shape[1], data.shape[2], data.shape[3])
    dset = h5_file.create_dataset('/%s' % key, data=data, chunks=chunks, compression='gzip', dtype='float32')

  h5_file.close()


#-------------------------------------------------------------------------------
# settings
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_path",type=str, default='', help='')
parser.add_argument("-s", "--setting",type=str, default='', help='')
parser.add_argument("-p", "--prefix",type=str, default='', help='')
args = parser.parse_args()

if len(args.prefix) > 0:
  args.prefix = '%s_' % args.prefix

args.d_invert = False
args.d_quantize = False
args.d_min = 0
args.d_max = 1
if args.setting == 'nmb32':
  args.out_dir = os.path.join(args.data_path, '..', 'patches', '%s%s' % (args.prefix, args.setting))
  args.n_patches_p_file = 65536
  args.ph = 32
  args.pw = 32
  args.d_invert = True
  args.d_quantize = True
  args.d_min = 70
  args.d_max = 218
  args.rgb_min = 0
  args.rgb_max = 255
elif args.setting == 'nmb48':
  args.out_dir = os.path.join(args.data_path, '..', 'patches', '%s%s' % (args.prefix, args.setting))
  args.n_patches_p_file = 65536
  args.ph = 48
  args.pw = 48
  args.d_invert = True
  args.d_quantize = True
  args.d_min = 70
  args.d_max = 218
  args.rgb_min = 0
  args.rgb_max = 255
elif args.setting == 'nmb64':
  args.out_dir = os.path.join(args.data_path, '..', 'patches', '%s%s' % (args.prefix, args.setting))
  args.n_patches_p_file = 65536
  args.ph = 64
  args.pw = 64
  args.d_invert = True
  args.d_quantize = True
  args.d_min = 70
  args.d_max = 218
  args.rgb_min = 0
  args.rgb_max = 255
elif args.setting == 'nmb128':
  args.out_dir = os.path.join(args.data_path, '..', 'patches', '%s%s' % (args.prefix, args.setting))
  args.n_patches_p_file = 16384
  args.ph = 128
  args.pw = 128
  args.d_invert = True
  args.d_quantize = True
  args.d_min = 70
  args.d_max = 218
  args.rgb_min = 0
  args.rgb_max = 255
elif args.setting == 'nmb512':
  args.out_dir = os.path.join(args.data_path, '..', 'patches', '%s%s' % (args.prefix, args.setting))
  args.n_patches_p_file = 1
  args.ph = 512
  args.pw = 512
  args.d_invert = True
  args.d_quantize = True
  args.d_min = 70
  args.d_max = 218
  args.rgb_min = 0
  args.rgb_max = 255
elif args.setting == 'tm48':
  args.out_dir = os.path.join(args.data_path, '..', 'patches', '%s%s' % (args.prefix, args.setting))
  args.n_patches_p_file = 65536
  args.ph = 48
  args.pw = 48
  args.d_invert = False
  args.d_quantize = False
  args.d_min = 400
  args.d_max = 1000
  args.rgb_min = 400
  args.rgb_max = 1000
elif args.setting == 'tm128':
  args.out_dir = os.path.join(args.data_path, '..', 'patches', '%s%s' % (args.prefix, args.setting))
  args.n_patches_p_file = 16384
  args.ph = 128
  args.pw = 128
  args.d_invert = False
  args.d_quantize = False
  args.d_min = 400
  args.d_max = 1000
  args.rgb_min = 400
  args.rgb_max = 1000
else:
  raise Exception('unknown setting "%s"' % args.setting)

args.files = glob(os.path.join(args.data_path, '*.h5'))



#-------------------------------------------------------------------------------
# create data
#-------------------------------------------------------------------------------

#create out dir
if not os.path.exists(args.out_dir):
  os.makedirs(args.out_dir)


patches = {}
n_patches = 0
n_files = 0
for path_idx, path in enumerate(args.files):
  print('process file %06d of %06d' % (path_idx+1, len(args.files)))

  data = {}

  # read data
  f = h5py.File(path, 'r')

  if 'rgb' in f:
    rgb = np.array(f['rgb'])
    rgb = rgb.mean(2)
    rgb_min, rgb_max = rgb.min(), rgb.max()
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    rgb = rgb * (args.rgb_max - args.rgb_min) + args.rgb_min
    data['rgb'] = rgb.reshape((1, rgb.shape[0], rgb.shape[1]))

  if 'depth' in f:
    depth = np.array(f['depth'])
    d_min, d_max = depth.min(), depth.max()
    depth = (depth - d_min) / (d_max - d_min)
    if args.d_invert:
      depth = 1 - depth
    depth = depth * (args.d_max - args.d_min) + args.d_min
    if args.d_quantize:
      depth = np.round(depth)
    data['depth'] = depth.reshape((1, depth.shape[0], depth.shape[1]))

  # data to patches
  hrange = range(0, depth.shape[0] - args.ph + 1, args.ph)
  wrange = range(0, depth.shape[1] - args.pw + 1, args.pw)
  for h in hrange:
    for w in wrange:
      for key in data:
        patch = data[key][:, h : h + args.ph, w : w + args.pw]
        if key not in patches:
          patches[key] = []
        patches[key].append(patch)

      n_patches += 1
      if n_patches >= args.n_patches_p_file:
        h5_out_path = os.path.join(args.out_dir, 'ph%d_pw%d_%06d.h5' % (args.ph, args.pw, n_files))
        h5_write(h5_out_path, patches)
        n_files += 1
        n_patches = 0
        for key in patches:
          patches[key] = []

if n_patches > 0:
  h5_out_path = os.path.join(args.out_dir, 'ph%d_pw%d_%06d.h5' % (args.ph, args.pw, n_files))
  h5_write(h5_out_path, patches)

plt.show()
