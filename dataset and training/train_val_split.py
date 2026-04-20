# Split between train and val folders

from pathlib import Path
import random
import os
import sys
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files',
                    required=True)
parser.add_argument('--train_pct', help='Ratio of images to go to train folder; \
                    the rest go to validation folder (example: ".8")',
                    default=.8)

args = parser.parse_args()

data_path = args.datapath
train_percent = float(args.train_pct)

if not os.path.isdir(data_path):
   print('Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again.')
   sys.exit(0)
if train_percent < .01 or train_percent > 0.99:
   print('Invalid entry for train_pct. Please enter a number between .01 and .99.')
   sys.exit(0)
val_percent = 1 - train_percent


input_image_path = os.path.join(data_path,'images')
input_label_path = os.path.join(data_path,'labels')

cwd = os.getcwd()
train_img_path = os.path.join(cwd,'data/train/images')
train_txt_path = os.path.join(cwd,'data/train/labels')
val_img_path = os.path.join(cwd,'data/validation/images')
val_txt_path = os.path.join(cwd,'data/validation/labels')

for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
   if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f'Created folder at {dir_path}.')


img_file_list = [path for path in Path(input_image_path).rglob('*')]
txt_file_list = [path for path in Path(input_label_path).rglob('*')]

print(f'Number of image files: {len(img_file_list)}')
print(f'Number of annotation files: {len(txt_file_list)}')

file_num = len(img_file_list)
train_num = int(file_num*train_percent)
val_num = file_num - train_num
print('Images moving to train: %d' % train_num)
print('Images moving to validation: %d' % val_num)

for i, set_num in enumerate([train_num, val_num]):
  for ii in range(set_num):
    img_path = random.choice(img_file_list)
    img_fn = img_path.name
    base_fn = img_path.stem
    txt_fn = base_fn + '.txt'
    txt_path = os.path.join(input_label_path,txt_fn)

    if i == 0:
      new_img_path, new_txt_path = train_img_path, train_txt_path
    elif i == 1: 
      new_img_path, new_txt_path = val_img_path, val_txt_path

    shutil.copy(img_path, os.path.join(new_img_path,img_fn))
    if os.path.exists(txt_path):
      shutil.copy(txt_path,os.path.join(new_txt_path,txt_fn))

    img_file_list.remove(img_path)
# example use in a cell: !python train_val_split.py --datapath="/content/custom_data" --train_pct=0.9 