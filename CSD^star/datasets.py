"""Return training and evaluation/test datasets from config files."""
# import jax
import tensorflow as tf
import tensorflow_datasets as tfds


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)

def get_EIT_dataset(config):
  # batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

  if config.data.dataset == 'EIT':
    import os
    from pprint import pprint
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    class EITdataset(Dataset):
      """
      return Gaussian-newton(condition), ground truth
      """

      def __init__(self, data_file_list):
        pprint(data_file_list)
        xs_all, xs_inv_all ,ys_all= None, None,None
        for data_file in data_file_list:
          if os.path.exists(data_file):
            d = np.load(data_file)
            xs_all = d['xs'] if (xs_all is None) else np.r_[xs_all, d['xs']]
            xs_inv_all = d['xs_gn'] if (xs_inv_all is None) else np.r_[xs_inv_all, d['xs_gn']]
            ys_all=d['ys'] if (ys_all is None) else np.r_[ys_all, d['ys']]
        # b, h, w = xs_all.shape

        self.xs = torch.from_numpy(xs_all).float().unsqueeze(axis=1)
        self.xs_inv = torch.from_numpy(xs_inv_all).float().unsqueeze(axis=1)
        self.ys = torch.from_numpy(ys_all).float()
        print('-' * 50)
        print(f'    xs:{self.xs.shape} \nxs_inv:{self.xs_inv.shape} \nys:{self.ys.shape}')
        print('-' * 50)

      def __getitem__(self, index):
        x_inv = self.xs_inv[index]
        x = self.xs[index]
        y = self.ys[index]
        return x_inv, x,y

      def __len__(self):
        return len(self.xs)


    def train_dataloader(train_data_file_list,batch_size):
      """
      Data loader for the training data.

      Returns
      -------
      DataLoader
          Training data loader.
      """
      dataset = EITdataset(train_data_file_list)
      dataloader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True, pin_memory=True, drop_last=True)
      print(
        f'TRAINING: \ndata length: {dataset.__len__()}\n batch_size: {dataloader.batch_size} \n iters/epoch: {len(dataloader)}\n')
      """
      x_inv,x = next(iter(dataset))
      fig, axes = plt.subplots(1,2)
      axes[0].imshow(x_inv[0])
      axes[1].imshow(x[0])
      plt.show()
      """
      return dataloader

    def val_dataloader(val_data_file_list,batch_size):
      """
      Data loader for the training data.

      Returns
      -------
      DataLoader
          Training data loader.
      """
      dataset = EITdataset(val_data_file_list)
      dataloader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=False, pin_memory=True,drop_last=True)
      print(
        f' VALIDATION: \n data length: {dataset.__len__()}\n batch_size: {dataloader.batch_size} \n iters/epoch: {len(dataloader)}\n')
      """
      x_inv,x = next(iter(dataset))
      fig, axes = plt.subplots(1,2)
      axes[0].imshow(x_inv[0])
      axes[1].imshow(x[0])
      plt.show()
      """
      return dataloader

    def test_dataloader(test_data_file_list):
      """
      Data loader for the training data.

      Returns
      -------
      DataLoader
          Training data loader.
      """
      dataset = EITdataset(test_data_file_list)
      dataloader = DataLoader(dataset,
                              batch_size=1,
                              num_workers=0,
                              shuffle=False, pin_memory=True)
      print(
        f' TEST: \n data length: {dataset.__len__()}\n batch_size: {dataloader.batch_size} \n iters/epoch: {len(dataloader)}\n')
      """
      x_inv,x = next(iter(dataset))
      fig, axes = plt.subplots(1,2)
      axes[0].imshow(x_inv[0])
      axes[1].imshow(x[0])
      plt.show()
      """
      return dataloader

    # change to your own data path
    train_data_file_list = [os.path.join(r'data_path', f'{i}.npz') for i in
                            range(0, 24)]
    val_data_file_list = [os.path.join(r'data_path', f'{i}.npz') for i in
                          range(25, 26)]
    test_data_file_list = [os.path.join(r'data_path', f'{i}.npz') for i in
                           range(26, 27)]

    train_ds = train_dataloader(train_data_file_list, config.training.batch_size)
    val_ds = val_dataloader(val_data_file_list,config.eval.batch_size)
    test_ds = test_dataloader(test_data_file_list)

    return train_ds, val_ds, test_ds