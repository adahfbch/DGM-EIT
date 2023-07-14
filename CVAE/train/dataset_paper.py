"""
dataset
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
import os

# change to your own data path
train_data_file_list = [os.path.join(r'data_path', f'{i}.npz') for i in
                    range(0, 24)]
val_data_file_list  = [os.path.join(r'data_path', f'{i}.npz') for i in
                    range(24, 27)]
test_data_file_list= [os.path.join(r'data_path', f'{i}.npz') for i in
                    range(27, 28)]

# path = r'data_path'
# train_data = [os.path.join(path,f'{i}.npz') for i in range(24)]
xs_train, ys_train = None,None
for data_file in train_data_file_list:
    if os.path.exists(data_file):
        d = np.load(data_file)
        xs_train = d['xs'] if (xs_train is None) else np.r_[xs_train, d['xs']]
        ys_train = d['ys'] if (ys_train is None) else np.r_[ys_train, d['ys']]

# val_data = np.load(r'data_path')
# xs_val = val_data['xs']
# ys_val = val_data['ys']

# val_data = [os.path.join(path, f'{i}.npz') for i in range(24,27)]
xs_val, ys_val = None,None
for data_file in val_data_file_list:
    if os.path.exists(data_file):
        d = np.load(data_file)
        xs_val = d['xs'] if (xs_val is None) else np.r_[xs_val, d['xs']]
        ys_val = d['ys'] if (ys_val is None) else np.r_[ys_val, d['ys']]


# test_data = np.load(r'data_path')
# xs_test = test_data['xs']
# ys_test = test_data['ys']

# test_data = [os.path.join(path, f'{i}.npz') for i in range(27,30)]
xs_test, ys_test, xs_gn_test = None, None, None
for data_file in test_data_file_list:
    if os.path.exists(data_file):
        d = np.load(data_file)
        xs_test = d['xs'] if (xs_test is None) else np.r_[xs_test, d['xs']]
        ys_test = d['ys'] if (ys_test is None) else np.r_[ys_test, d['ys']]
        xs_gn_test = d['xs_gn'] if (xs_gn_test is None) else np.r_[xs_gn_test, d['xs_gn']]

# test_data = np.load(r'data_path')
# xs_test = test_data['xs']
# ys_test = test_data['ys']
# xs_gn_test = test_data['xs_gn']

xs_train = xs_train.astype(np.float32)
ys_train = ys_train.astype(np.float32)
xs_val = xs_val.astype(np.float32)
ys_val = ys_val.astype(np.float32)
xs_test = xs_test.astype(np.float32)
ys_test = ys_test.astype(np.float32)
xs_gn_test = xs_gn_test.astype(np.float32)

transform = transforms.Compose([transforms.ToTensor()])

xs_train = transform(xs_train)
xs_train = xs_train.view(1, *xs_train.size())
xs_train = xs_train.permute(2, 0, 1, 3)

ys_train = transform(ys_train)
ys_train = ys_train.permute(1, 0, 2)

xs_val = transform(xs_val)
xs_val = xs_val.view(1, *xs_val.size())
xs_val = xs_val.permute(2, 0, 1, 3)


ys_val = transform(ys_val)
ys_val = ys_val.permute(1, 0, 2)

xs_test = transform(xs_test)
xs_test = xs_test.view(1, *xs_test.size())
xs_test = xs_test.permute(2, 0, 1, 3)

ys_test = transform(ys_test)
ys_test = ys_test.permute(1, 0, 2)

xs_gn_test = transform(xs_gn_test)
xs_gn_test = xs_gn_test.view(1, *xs_gn_test.size())
xs_gn_test = xs_gn_test.permute(2, 0, 1, 3)

dd_train = [xs_train,ys_train]
dd_val = [xs_val,ys_val]