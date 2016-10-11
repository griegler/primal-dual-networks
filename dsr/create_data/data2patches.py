#!/usr/bin/env python2


import numpy as np
import h5py
import sys
import time
import argparse
import os
import math

# 148x148 - min_std 50
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", nargs='*', type=str, default=[''])
parser.add_argument("--ph", type=int, default=32)
parser.add_argument("--pw", type=int, default=-1)
parser.add_argument("--stride", type=int, default=-1)
parser.add_argument("--min_std", type=float, default=5.0)
parser.add_argument("--plot", type=bool, default=False)
parser.add_argument("--in_depth", type=bool, default=False)
args = parser.parse_args()

if args.pw <= 0:
  args.pw = args.ph
if args.stride <= 0:
  args.stride = args.ph
if args.plot:
  from mpl_toolkits.mplot3d.axes3d import Axes3D
  import seaborn as sns
  import matplotlib.pyplot as plt
  sns.set(style="whitegrid", palette="muted", color_codes=True)

# load data
for idx, h5_path in enumerate(sorted(args.files)):
  h5_file = h5py.File(h5_path)
  imgs_ta = np.array(h5_file['/ta_depth'])
  if args.in_depth:
    imgs_in = np.array(h5_file['/in_depth'])
  h5_file.close()

  if len(imgs_ta.shape) != 4:
    raise Exception('invalid shape size')

  hrange = range(0, imgs_ta.shape[2] - args.ph + 1, args.stride)
  wrange = range(0, imgs_ta.shape[3] - args.pw + 1, args.stride)

  n_patches_h = len(hrange)
  n_patches_w = len(wrange)
  n_patches = n_patches_h * n_patches_w
  patches_ta = np.ndarray(shape=(imgs_ta.shape[0] * n_patches, imgs_ta.shape[1], args.ph, args.pw), dtype=float)
  if args.in_depth:
    patches_in = np.ndarray(shape=(imgs_in.shape[0] * n_patches, imgs_in.shape[1], args.ph, args.pw), dtype=float)

  # collect patches_ta
  patch_idx = 0
  for n in range(imgs_ta.shape[0]):
    for h in hrange:
      for w in wrange:
        # print(h, w, imgs_ta.shape[2], imgs_ta.shape[3])
        patch_ta = imgs_ta[n, :, h : h + args.ph, w : w + args.pw]
        patches_ta[patch_idx] = patch_ta
        if args.in_depth:
          patch_in = imgs_in[n, :, h : h + args.ph, w : w + args.pw]
          patches_in[patch_idx] = patch_in

        # plt.imshow(patches_ta[patch_idx][0], interpolation='nearest', cmap='YlGnBu')
        # plt.draw()
        # plt.waitforbuttonpress()

        patch_idx += 1

  stds = patches_ta.reshape((patches_ta.shape[0], -1)).std(axis=1)

  patches_all = patches_ta
  patches_ta = patches_ta[stds >= args.min_std]
  if args.in_depth:
    patches_in = patches_in[stds >= args.min_std]
  print('%d/%d - reduced #patches_ta from %d to %d' % (idx+1, len(args.files), patches_all.shape[0], patches_ta.shape[0]))

  if args.plot and idx < 1:
    plt.figure(idx)
    f, axes = plt.subplots(1, 2, sharey=True, num=idx)
    sns.distplot(stds, ax=axes[0])
    sns.distplot(stds[stds >= args.min_std], ax=axes[1])
    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress()

  # generate out path
  h5_dir = os.path.dirname(h5_path)
  h5_base, h5_ext = os.path.splitext(os.path.basename(h5_path))
  h5_out_path = os.path.join(h5_dir, 'ph%d_pw%d_%s%s' % (args.ph, args.pw, h5_base, h5_ext))
  print(h5_out_path)

  # write h5 path
  h5_file = h5py.File(h5_out_path, 'w')
  chunks=(min(patches_ta.shape[0], 1000), patches_ta.shape[1], patches_ta.shape[2], patches_ta.shape[3])
  dset = h5_file.create_dataset('/ta_depth', data=patches_ta, chunks=chunks, compression='gzip', dtype='float32')
  if args.in_depth:
    dset = h5_file.create_dataset('/in_depth', data=patches_in, chunks=chunks, compression='gzip', dtype='float32')
  h5_file.close()

if args.plot:
  plt.show()
