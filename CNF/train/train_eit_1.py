"""
2022年6月6日 添加注释
            原始代码可参考 https://github.com/jleuschn/cinn_for_imaging
"""
import os
import sys
from warnings import warn

sys.path.append('../')
# sys.path.append('../../')
import numpy as np
import platform
from pprint import pprint
import matplotlib.pyplot as plt
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from dival.measure import PSNR, SSIM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from util.torch_losses import CINNNLLLoss
from reconstructors.networks.layers import Flatten
from reconstructors.networks.layers import InvertibleDownsampling, Fixed1x1ConvOrthogonal, Split
from torchsummary import summary
from torchvision import utils as vutils
import imlib as im

# %
class ResNetCondNet(nn.Module):
    """
    Conditional network H that sits on top of the invertible architecture. It
    features a FBP operation at the beginning and continues with post-
    processing steps.
    Attributes
    ----------
    resolution_levels : torch module list
        Building blocks of the conditional network.
    Methods
    -------
    forward(c)
        Compute the forward pass.
    """

    def __init__(self, img_size,
                 downsample_levels=5,
                 cond_conv_channels=[4, 16, 32, 64, 64, 32],
                 use_fc_block=True,
                 cond_fc_size=128,
                 condition_input_type='x_inv'):

        """
        Parameters
        ----------
        condition_input_type 'data' or 'x_inv'
        img_size : TYPE
            DESCRIPTION.
        fc_cond_dim : int, optional
            DESCRIPTION.
        filter_type : TYPE, optional
            DESCRIPTION. The default is 'Hann'.
        frequency_scaling : TYPE, optional
            DESCRIPTION. The default is 1..
        Returns
        -------
        None.
        """
        super().__init__()

        self.img_size = img_size
        self.dsl = downsample_levels

        # FBP and resizing layers
        self.unet_out_shape = 16  # ？？

        self.img_size = img_size
        self.condition_input_type = condition_input_type
        self.dsl = downsample_levels
        self.fc_cond_dim = cond_fc_size
        self.shapes = [self.unet_out_shape] + cond_conv_channels  # 在cond_conv_channels的基础上加了一个维度，维度大小为unet_out_shape
        self.use_fc_block = use_fc_block
        levels = []
        for i in range(self.dsl):
            levels.append(self.create_subnetwork(ds_level=i))
        if self.use_fc_block:
            levels.append(self.create_subnetwork_fc(ds_level=self.dsl))

        self.preprocessing_net = nn.Sequential(  # 这一步的目的是什么：从结果来看iuput的维度增加了
            ResBlock(in_ch=1, out_ch=8),
            ResBlock(in_ch=8, out_ch=8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1),
            ResBlock(in_ch=8, out_ch=16),
            ResBlock(in_ch=16, out_ch=16),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            ResBlock(in_ch=16, out_ch=self.unet_out_shape),

        )
        self.resolution_levels = nn.ModuleList(levels)

    def forward(self, c):
        """
        xxx 对lodopab数据,outputs= [1,2,..,6]  i.shape=[1, 4, 192, 192]
        Computes the forward pass of the conditional network and returns the
        results of all building blocks in self.resolution_levels.
        Parameters
        ----------
        c : torch tensor
            Input to the conditional network (measurement).
        Returns
        -------
        List of torch tensors
            Results of each block of the conditional network.
        """

        outputs = []
        if self.condition_input_type == 'data':  # 条件输入是观测数据，那么解inverse problem（numerical or dnn)
            c = self.solve_eit_layer(c)
            shape_cond = c.shape
            c = c.view(shape_cond[0], 1, self.img_size[0], self.img_size[1])

        c_unet = self.preprocessing_net(c)
        for m in self.resolution_levels:
            # print(m(c).shape)
            outputs.append(m(c_unet))
        return outputs

    def create_subnetwork(self, ds_level):  # 应该是在定义 downsampling network
        padding = [2, 2, 2, 2, 1, 1, 1]
        kernel = [5, 5, 5, 5, 3, 3]

        modules = []

        for i in range(ds_level + 1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i],
                                     out_channels=self.shapes[i + 1],
                                     kernel_size=kernel[i],
                                     padding=padding[i],
                                     stride=2))

            modules.append(ResBlock(in_ch=self.shapes[i + 1], out_ch=self.shapes[i + 1]))
            # modules.append(nn.BatchNorm2d(self.shapes[i+1]))
            # modules.append(nn.LeakyReLU())

            # modules.append(nn.Conv2d(in_channels=self.shapes[i+1],
            #                out_channels=self.shapes[i+1],
            #                kernel_size=3,
            #                padding=1,
            #                stride=1))
            # modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level + 1],
                                 out_channels=self.shapes[ds_level + 1],
                                 kernel_size=1))
        return nn.Sequential(*modules)

    def create_subnetwork_fc(self, ds_level):
        padding = [2, 2, 2, 2, 1, 1, 1]
        kernel = [5, 5, 5, 5, 3, 3]

        modules = []

        for i in range(ds_level + 1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i],
                                     out_channels=self.shapes[i + 1],
                                     kernel_size=kernel[i],
                                     padding=padding[i],
                                     stride=2))

            modules.append(ResBlock(in_ch=self.shapes[i + 1], out_ch=self.shapes[i + 1]))

            # modules.append(nn.BatchNorm2d(self.shapes[i+1]))
            # modules.append(nn.LeakyReLU())

            # modules.append(nn.Conv2d(in_channels=self.shapes[i+1],
            #                out_channels=self.shapes[i+1],
            #                kernel_size=3,
            #                padding=1,
            #                stride=1))
            # modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level + 1],
                                 out_channels=self.shapes[ds_level + 1],
                                 kernel_size=1))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level + 1],
                                 out_channels=self.fc_cond_dim,
                                 kernel_size=1))
        modules.append(nn.AvgPool2d(6, 6))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.fc_cond_dim))

        return nn.Sequential(*modules)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, size=3):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, 2 * out_ch, size, padding=int(size / 2), stride=1),
            nn.BatchNorm2d(2 * out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(2 * out_ch, 2 * out_ch, size, padding=int(size / 2), stride=1),
            nn.BatchNorm2d(2 * out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(2 * out_ch, out_ch, 1, padding=0, stride=1))

        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1)

        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.final_activation = nn.LeakyReLU()

    def forward(self, x):
        conv = self.conv_block(x)

        res = self.residual(x)

        y = self.batch_norm(conv + res)
        y = self.final_activation(y)

        return y


def subnet_conv3x3(in_ch, out_ch):
    """
    Sub-network with 3x3 2d-convolutions and leaky ReLU activation.
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    Returns
    -------
    torch sequential model
        The sub-network.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, 2 * in_ch, 3, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(2 * in_ch, 2 * in_ch, 3, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(2 * in_ch, out_ch, 3, padding=1))


def subnet_conv1x1(in_ch, out_ch):
    """
    Sub-network with 1x1 2d-convolutions and leaky ReLU activation.
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    Returns
    -------
    torch sequential model
        The sub-network.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, 2 * in_ch, 1),
        nn.LeakyReLU(),
        nn.Conv2d(2 * in_ch, 2 * in_ch, 1),
        nn.LeakyReLU(),
        nn.Conv2d(2 * in_ch, out_ch, 1))


def subnetUncond(in_ch, out_ch):
    """
    Sub-netwok with 1x1 2d-convolutions for unconditioned parts of the cINN.
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    Returns
    -------
    torch sequential model
        The sub-network.
    """
    # print(in_ch,out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, 2 * in_ch, 1),
        nn.LeakyReLU(),
        nn.Conv2d(2 * in_ch, 2 * in_ch, 1),
        nn.LeakyReLU(),
        nn.Conv2d(2 * in_ch, out_ch, 1)
    )


def subnet_fc(in_ch, out_ch):
    """
    Sub-network with fully connected layers and leaky ReLU activation.
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    Returns
    -------
    torch sequential model
        The sub-network.
    """
    return nn.Sequential(nn.Linear(in_ch, 2 * in_ch),
                         nn.LeakyReLU(),
                         nn.Linear(2 * in_ch, 2 * in_ch),
                         nn.LeakyReLU(),
                         nn.Linear(2 * in_ch, out_ch))


def _add_conditioned_section(nodes, downsampling_level, num_blocks, cond, coupling, act_norm, permutation,
                             clampling=2.5):
    """
    Add conditioned notes to the network.
    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsampling_level: int
        Current downsampling level
    num_blocks: int
        Number of coupling blocks
    cond : TYPE
        FrEIA condition note
    coupling: str
        Type of coupling used
    act_norm: bool
        whether to use act norm
    permutation: str
        which permutation to use
    clamping: float
        clamping for glow coupling layer
    Returns
    -------
    None.
    """

    for k in range(num_blocks):
        if k % 2 == 0:
            subnet = subnet_conv1x1
        else:
            subnet = subnet_conv3x3

        if coupling == 'affine':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': clampling},
                                 conditions=cond,
                                 name="GLOWBlock_{}.{}".format(downsampling_level, k)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                                 {'subnet_constructor': subnet},
                                 conditions=cond,
                                 name="NICEBlock_{}.{}".format(downsampling_level, k)))

        if act_norm:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, name="ActNorm_{}.{}".format(downsampling_level, k)))

        if permutation == "1x1":
            nodes.append(Ff.Node(nodes[-1].out0, Fixed1x1ConvOrthogonal,
                                 {},
                                 name='1x1Conv_{}.{}'.format(downsampling_level, k)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom,
                                 {'seed': (k + 1) * (downsampling_level + 1)},
                                 name='PermuteRandom_{}.{}'.format(downsampling_level, k)))


def _add_downsample(nodes, downsample, coupling, clamping=2.5, use_act_norm=True):
    """
    Downsampling operations.
    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsample : str
        Downsampling method. Currently there are three options: 'haar', 'reshape' and 'invertible'.
    in_ch : int
        Number of input channels.
    clamping : float, optional
        The default value is 1.5.
    Returns
    -------
    None.
    """

    if downsample == 'haar':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling,
                             {'rebalance': 0.5, 'order_by_wavelet': True},
                             name='haar'))
    if downsample == 'reshape':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.IRevNetDownsampling, {},
                             name='reshape'))
    if downsample == 'invertible':
        nodes.append(Ff.Node(nodes[-1].out0, InvertibleDownsampling,
                             {'stride': 2, 'method': 'cayley', 'init': 'haar',
                              'learnable': True}, name='invertible'))

    for i in range(2):
        if coupling == 'affine':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnetUncond,
                                  'clamp': clamping}))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                                 {'subnet_constructor': subnetUncond}))
        if use_act_norm:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}))


def total_variation(img):
    r"""Function that computes Total Variation according to [1].
    Args:
        img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.
    Return:
         a scalar with the computer loss.
    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)
    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       total_variation_denoising.html>`__.
    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 3 or len(img.shape) > 4:
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)

    return res1 + res2


class EITdataset(Dataset):
    """
    return Gaussian-newton(condition), ground truth
    """

    def __init__(self, data_file_list, target_size):
        pprint(data_file_list)
        xs_all, xs_inv_all = None, None
        for data_file in data_file_list:
            if os.path.exists(data_file):
                d = np.load(data_file)
                xs_all = d['xs'] if (xs_all is None) else np.r_[xs_all, d['xs']]
                xs_inv_all = d['xs_gn'] if (xs_inv_all is None) else np.r_[xs_inv_all, d['xs_gn']]
        # b, h, w = xs_all.shape
        self.xs = torch.from_numpy(xs_all).float().unsqueeze(axis=1)
        self.xs_inv = torch.from_numpy(xs_inv_all).float().unsqueeze(axis=1)
        # if h != target_size and target_size < h:
        #     left_top = (h - target_size) // 2
        #     self.xs = torch.from_numpy(
        #         xs_all[:, left_top:left_top + target_size, left_top:left_top + target_size]).float().unsqueeze(axis=1)
        #     self.xs_inv = torch.from_numpy(
        #         xs_inv_all[:, left_top:left_top + target_size, left_top:left_top + target_size]).float().unsqueeze(
        #         axis=1)
        print('-' * 50)
        print(f'    xs:{self.xs.shape} \nxs_inv:{self.xs_inv.shape}')
        print('-' * 50)

    def __getitem__(self, index):
        x_inv = self.xs_inv[index]
        x = self.xs[index]
        return x_inv, x

    def __len__(self):
        return len(self.xs)


class EITDataModule(pl.LightningDataModule):
    def __init__(self, train_data_file_list,
                 val_data_file_list,
                 test_data_file_list,
                 target_size: int = 128,
                 batch_size: int = 4,
                 num_data_loader_workers: int = 0):
        super().__init__()
        self.train_data_file_list = train_data_file_list
        self.val_data_file_list = val_data_file_list
        self.test_data_file_list = test_data_file_list
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_data_loader_workers = num_data_loader_workers

    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.
        """
        dataset = EITdataset(self.train_data_file_list, target_size=self.target_size)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_data_loader_workers,
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

    def val_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.
        """
        dataset = EITdataset(self.val_data_file_list, target_size=self.target_size)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size*4,
                                num_workers=self.num_data_loader_workers,
                                shuffle=False, pin_memory=True)
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

    def test_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.
        """
        dataset = EITdataset(self.test_data_file_list, target_size=self.target_size)
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=self.num_data_loader_workers,
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


def check_img_size(H, W, downsample_levels):
    found_shape = False
    img_size = (H, W)
    while not found_shape:
        if (sum([img_size[0] % 2 ** (i + 1) for i in range(downsample_levels)]) +
            sum([img_size[1] % 2 ** (i + 1) for i in range(downsample_levels)])
        ) == 0:
            found_shape = True
        else:
            found_shape = False
            img_size = (img_size[0] + 1, img_size[1] + 1)
    return img_size


def Re_sigma(x, reco_eit):
    Re_up = torch.nn.L1Loss()(x, reco_eit)
    t = torch.zeros(x.shape).cpu()
    Re_down = torch.nn.L1Loss()(x, t)
    Re_sigma = Re_up / Re_down
    return Re_sigma


def DR(x, reco_eit):
    DR_up = torch.max(reco_eit) - torch.min(reco_eit)
    DR_down = torch.max(x) - torch.min(x)
    DR = DR_up / DR_down
    return DR


# %  主要部分
class CINN(pl.LightningModule):
    def __init__(self, img_size,
                 in_ch=1,  # 我理解的是输入通道数
                 downsample_levels=5,  # 这个downsample_levels是什么意思
                 cond_conv_channels=[4, 16, 32, 64, 64, 32],  # 控制conditional net中尺寸的变换
                 downsampling: str = 'invertible',  # 采用“可逆降采样”方法
                 coupling: str = 'affine',  # 使用仿射耦合
                 num_fc: int = 4,  # 全连接层的层数
                 clamping: float = 2.5,  # 不清楚
                 use_fc_block: bool = True,  # 是否使用全连接层
                 cond_fc_size: int = 64,  # conditional net全连接层的shape
                 use_act_norm: bool = True,  #
                 num_blocks: int = 6,  # 耦合次数
                 permutation: str = '1x1',  # 使用1*1卷积进行permutation
                 add_reg: float = 0):  # =0 无正则化惩罚项目，>0表示惩罚性系数
        super().__init__()

        self.permutation = permutation
        self.add_reg = add_reg
        self.img_size = img_size
        self.num_blocks = num_blocks
        self.use_act_norm = use_act_norm
        self.downsampling = downsampling
        self.cond_fc_size = cond_fc_size
        self.downsample_levels = downsample_levels
        self.cond_conv_channels = cond_conv_channels
        self.use_fc_block = use_fc_block
        self.coupling = coupling
        self.num_fc = num_fc
        self.in_ch = in_ch
        self.clamping = clamping
        self.train_noise = (0., 0.005)
        self.data_range = [1e-05, 2.5]
        self.criterion = CINNNLLLoss(distribution='normal')  # NLLLoss = negative log likelihood loss
        self.mse_loss = torch.nn.MSELoss()
        self.cond_net = ResNetCondNet(img_size=self.img_size,  # 残差调节网络
                                      downsample_levels=self.downsample_levels,
                                      cond_conv_channels=self.cond_conv_channels,
                                      use_fc_block=self.use_fc_block,
                                      cond_fc_size=self.cond_fc_size)
        # build the cINN
        self.cinn = self.build_inn()

        # initialize the values of the parameters
        self.init_params()

    def build_inn(self):
        """
        Connect the building blocks of the cINN.
        Returns
        -------
        FrEIA ReversibleGraphNet
            cINN model.
        """

        # initialize lists for the split and conditioning notes
        split_nodes = []
        conditions = []

        # create the conditioning notes
        for i in range(self.downsample_levels):
            conditions.append(Ff.ConditionNode(self.cond_conv_channels[i],
                                               self.img_size[0] / (2 ** (i + 1)),
                                               self.img_size[1] / (2 ** (i + 1)),
                                               name="cond_{}".format(i)))
        if self.use_fc_block:
            conditions.append(Ff.ConditionNode(self.cond_fc_size, name="cond_{}".format(self.downsample_levels)))

        ### build the network & add conditioning and splits ###

        ## 1) Input region + first downsampling (1 x 1 x 1 -> 4 x 1/2 x 1/2)
        nodes = [Ff.InputNode(self.in_ch, self.img_size[0], self.img_size[1], name='inp')]

        _add_downsample(nodes, self.downsampling, coupling=self.coupling, use_act_norm=self.use_act_norm)

        for downsample_step in range(self.downsample_levels - 1):
            _add_conditioned_section(nodes,
                                     downsampling_level=downsample_step,
                                     cond=conditions[downsample_step],
                                     num_blocks=self.num_blocks,
                                     coupling=self.coupling,
                                     act_norm=self.use_act_norm,
                                     permutation=self.permutation)

            _add_downsample(nodes, self.downsampling, coupling=self.coupling, use_act_norm=self.use_act_norm)

            nodes.append(Ff.Node(nodes[-1], Split,
                                 {'n_sections': 2, 'dim': 0},
                                 name="split_{}".format(downsample_step)))
            split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                                       name='flatten_split_{}'.format(downsample_step)))

        # c) flatten the output from the previous layer
        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='flatten'))

        if self.use_fc_block:
            nodes.append(Ff.Node(nodes[-1], Split,
                                 {'section_sizes': [128], 'dim': 0, 'n_sections': None},
                                 name="split_fc"))
            split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                                       name='flatten_split_fc'))
            ## 4) Random Permute -> Fully Connected Cond
            for k in range(self.num_fc):
                nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed': k},
                                     name='Permute_{}'.format(k)))

                if self.coupling == 'affine':
                    nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                                         {'subnet_constructor': subnet_fc,
                                          'clamp': self.clamping},
                                         conditions=conditions[-1],
                                         name='GlowBlock_fc_{}_{}'.format(
                                             self.downsample_levels + 1, k)))
                else:
                    nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                                         {'subnet_constructor': subnet_fc},
                                         conditions=conditions[-1],
                                         name='NICEBlock_fc_{}_{}'.format(
                                             self.downsample_levels + 1, k)))

        ## 5) concat all split notes and network output
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim': 0}, name='concat_splits'))

        nodes.append(Ff.OutputNode(nodes[-1], name='out'))

        return Ff.GraphINN(nodes + conditions + split_nodes, verbose=False)

    def init_params(self):
        """
        Initialize the parameters of the model.
        Returns
        -------
        None.
        """
        # approx xavier
        # for p in self.cond_net.parameters():
        #    p.data = 0.02 * torch.randn_like(p)
        for key, param in self.cinn.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.02 * torch.randn(param.data.shape)
                # last convolution in the coeff func
                if len(split) > 3 and split[3][-1] == '4':
                    param.data.fill_(0.)

    def forward(self, cinn_input, cond_input, rev: bool = True, cut_ouput: bool = True):
        """
        xxx  z的大小是147456=384**2. 虽然 z=(z^0,..,z^l,..,z^L) 但是用forward不能确定L和z^l的大小
        xxx  测试方法 z, log_jac =
        self.cinn(torch.randn((2,1,362,362), torch.randn((2,1,1000,513), rev=False) batch_size>=2
        Inference part of the whole model. There are two directions of the
        cINN. These are controlled by rev:
            rev==True:  Create a reconstruction x for a random sample z
                        and the conditional measurement y (Z|Y) -> X.
            rev==False: Create a sample z from a reconstruction x
                        and the conditional measurement y (X|Y) -> Z .
        Parameters
        ----------
        cinn_input : torch tensor
            Input to the cINN model. Depends on rev:
                rev==True: Random vector z.
                rev==False: Reconstruction x.
        cond_input : torch tensor
            Input to the conditional network. This is the measurement y.
        rev : bool, optional
            Direction of the cINN flow. For True it is Z -> X to create a
            single reconstruction. Otherwise X -> Z.
            The default is True.
        cut_ouput : bool, optional
            Cut the output of the network to the domain size of the operator.
            This is only relevant if rev==True.
            The default is True.
        Returns
        -------
        torch tensor or tuple of torch tensor
            rev==True:  x : Reconstruction
            rev==False:
                    z : Sample from the target distribution
 log_jac(log_jac_det) : log det of the Jacobian  log |det dz/dx | <==== Eq.(17&18)
        """
        # direction (Z|Y) -> X
        if rev:
            x, _ = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev)
            if cut_ouput:
                return x[:, :, :self.img_size[0], :self.img_size[1]]
            else:
                return x
        # direction (X|Y) -> Z
        else:
            z, log_jac = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev)
            return z, log_jac

    def training_step(self, batch, batch_idx):
        """
        Pytorch Lightning training step. Should be independent of forward()
        according to the documentation. The loss value is logged.
        Parameters
        ----------
        batch : tuple of tensor
            Batch of measurement y and ground truth reconstruction gt.
        batch_idx : int
            Index of the batch.
        Returns
        -------
        result : TYPE
            Result of the training step.
        """
        y, gt = batch  # (b,208)or(b,1,256,256),   (b,1,256,256)

        # run the conditional network
        c = self.cond_net(y)

        # Sample noise from Normal(train_noise[0], train_noise[1])
        if self.train_noise[1] > 0:
            cinn_input = gt + torch.randn((gt.shape), device=self.device) * self.train_noise[1] + self.train_noise[0]
            if self.data_range is not None:
                cinn_input = torch.clip(cinn_input, min=self.data_range[0], max=self.data_range[1])
        else:
            cinn_input = gt

        # run the cINN from X -> Z with the gt data and conditioning
        zz, log_jac = self.cinn(cinn_input, c)

        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)

        if self.add_reg:
            x = self.forward(zz, y, rev=True, cut_ouput=True)
            tv_loss = total_variation(x)
            self.log('nll_loss', loss)
            self.log('tv_loss', tv_loss.mean())
            loss = loss + self.add_reg * tv_loss.mean()

        # Log the training loss
        self.log('train_loss', loss)

        self.last_batch = batch
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Pytorch Lightning validation step. Should be independent of forward()
        according to the documentation. The loss value is logged and the
        best model according to the loss (lowest) checkpointed.
        Parameters
        ----------
        batch : tuple of tensor
            Batch of measurement y and ground truth reconstruction gt.
        batch_idx : int
            Index of the batch.
        Returns
        -------
        result : TYPE
            Result of the validation step.
        """
        y, gt = batch
        # run the conditional network
        c = self.cond_net(y)
        # run the cINN from X -> Z with the gt data and conditioning
        zz, log_jac = self.cinn(gt, c)

        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)

        # checkpoint the model and log the loss
        self.log('val_loss', loss)
        self.logger.experiment.add_scalar('val_loss_epoch', loss, self.current_epoch)

        self.last_batch = batch
        return loss

    def validation_epoch_end(self, result):
        """
        tensorboard --logdir=version_10
        no logging of histogram. Checkpoint gets big
        for name,params in self.named_parameters():
             self.logger.experiment.add_histogram(name, params, self.current_epoch)
        """
        y, gt = self.last_batch  # xxx 就是数据集的最后一个batch，为了易看，可以减小
        num_show = 7
        y, gt = y[:num_show, ...], gt[:num_show, ...]

        # xmean = 0
        # xstd = 0

        z = torch.randn((gt.shape[0], self.img_size[0] * self.img_size[1]), device=self.device)  # （B,H*W)
        with torch.no_grad():
            # reco, reco_std = self.reconstruct(y, return_std=True)

            x = self.forward(z, y, rev=True, cut_ouput=True)

            psnr_mean = np.mean([PSNR(xi.cpu(), gti.cpu()) for (xi, gti) in zip(x, gt)])
            ssim_mean = np.mean([SSIM(xi[0].cpu().numpy(), gti[0].cpu().numpy()) for (xi, gti) in zip(x, gt)])
            mse_mean = np.mean([((xi.cpu() - gti.cpu()) ** 2).mean() for (xi, gti) in zip(x, gt)])
            re_mean = np.mean([Re_sigma(gti.cpu(), xi.cpu()) for (xi, gti) in zip(x, gt)])
            ae_mean = np.mean([torch.nn.L1Loss()(xi.cpu(), gti.cpu()) for (xi, gti) in zip(x, gt)])
            dr_mean = np.mean([DR(gti.cpu(), xi.cpu()) for (xi, gti) in zip(x, gt)])
            # metrics = {'psnr': psnr_mean, 'ssim': ssim_mean, 'mse': mse_mean, 're': re_mean, 'ae': ae_mean, 'dr': dr_mean}
            # self.log_dict(metrics)
            # self.logger.experiment.add_image("reco_std", xstd, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('psnr', psnr_mean, self.current_epoch)
            self.logger.experiment.add_scalar('ssim', ssim_mean, self.current_epoch)
            self.logger.experiment.add_scalar('mse', mse_mean, self.current_epoch)
            self.logger.experiment.add_scalar('re', re_mean, self.current_epoch)
            self.logger.experiment.add_scalar('ae', ae_mean, self.current_epoch)
            self.logger.experiment.add_scalar('dr', dr_mean, self.current_epoch)


            imgs = torch.cat([gt, y, torch.clamp(x, 1e-05,2.5)], axis=0)
            # imgs = torch.cat([gt, y, x], axis=0)
            grid = torchvision.utils.make_grid(imgs, nrow=num_show)
            self.logger.experiment.add_image("a-truth-gn-reconstructions", grid, global_step=self.current_epoch,
                                             dataformats='CHW')
            # vutils.save_image(grid[0], './a-truth-gn-reconstructions/test.jpg')

            # gt = gt.squeeze()
            # y = y.squeeze()
            # self.logger.experiment.add_image("ground-truth", gt, global_step=self.current_epoch)
            # self.logger.experiment.add_image("generate-image", y, global_step=self.current_epoch)

            # conds = self.cond_net(y)  # ??? shapes of y and conds
            # for i, c in enumerate(conds):
            #     c = c.view(-1, 1, c.shape[-2], c.shape[-1])
            #     c_grid = torchvision.utils.make_grid(c, scale_each=True, normalize=True)
            #     self.logger.experiment.add_image(f"cond_level_{i}", c_grid, global_step=self.current_epoch)

    # def test_step(self, batch, batch_idx):
    #     """
    #     Pytorch Lightning validation step. Should be independent of forward()
    #     according to the documentation. The loss value is logged and the
    #     best model according to the loss (lowest) checkpointed.
    #     Parameters
    #     ----------
    #     batch : tuple of tensor
    #         Batch of measurement y and ground truth reconstruction gt.
    #     batch_idx : int
    #         Index of the batch.
    #     Returns
    #     -------
    #     result : TYPE
    #         Result of the validation step.
    #     """
    #     y, gt = batch
    #     # run the conditional network
    #     c = self.cond_net(y)
    #     # run the cINN from X -> Z with the gt data and conditioning
    #     zz, log_jac = self.cinn(gt, c)
    #
    #     # evaluate the NLL loss
    #     loss = self.criterion(zz=zz, log_jac=log_jac)
    #
    #     # checkpoint the model and log the loss
    #     self.log('test_loss', loss)
    #
    #     self.last_batch = batch
    #     return loss
    #
    # def test_epoch_end(self, result):
    #     """
    #     tensorboard --logdir=version_10
    #     no logging of histogram. Checkpoint gets big
    #     for name,params in self.named_parameters():
    #          self.logger.experiment.add_histogram(name, params, self.current_epoch)
    #     """
    #     y, gt = self.last_batch  # xxx 就是数据集的最后一个batch，为了易看，可以减小
    #     num_show = 4
    #     y, gt = y[:num_show, ...], gt[:num_show, ...]
    #
    #     # xmean = 0
    #     # xstd = 0
    #
    #     z = torch.randn((gt.shape[0], self.img_size[0] * self.img_size[1]), device=self.device)  # （B,H*W)
    #     with torch.no_grad():
    #         # reco, reco_std = self.reconstruct(y, return_std=True)
    #
    #         x = self.forward(z, y, rev=True, cut_ouput=True)
    #
    #         psnr_mean = np.mean([PSNR(xi.cpu(), gti.cpu()) for (xi, gti) in zip(x, gt)])
    #         ssim_mean = np.mean([SSIM(xi[0].cpu().numpy(), gti[0].cpu().numpy()) for (xi, gti) in zip(x, gt)])
    #         mse_mean = np.mean([torch.nn.MSELoss()(xi.cpu(), gti.cpu()) for (xi, gti) in zip(x, gt)])
    #         re_mean = np.mean([Re_sigma(gti.cpu(), xi.cpu()) for (xi, gti) in zip(x, gt)])
    #         ae_mean = np.mean([torch.nn.L1Loss()(xi.cpu(), gti.cpu()) for (xi, gti) in zip(x, gt)])
    #         dr_mean = np.mean([DR(gti.cpu(), xi.cpu()) for (xi, gti) in zip(x, gt)])
    #         psnr_var = np.var([PSNR(xi.cpu(), gti.cpu()) for (xi, gti) in zip(x, gt)])
    #         ssim_var = np.var([SSIM(xi[0].cpu().numpy(), gti[0].cpu().numpy()) for (xi, gti) in zip(x, gt)])
    #         mse_var = np.var([((xi.cpu() - gti.cpu()) ** 2).mean() for (xi, gti) in zip(x, gt)])
    #         re_var = np.var([Re_sigma(gti.cpu(), xi.cpu()) for (xi, gti) in zip(x, gt)])
    #         ae_var = np.var([torch.nn.L1Loss()(xi.cpu(), gti.cpu()) for (xi, gti) in zip(x, gt)])
    #         dr_var = np.var([DR(gti.cpu(), xi.cpu()) for (xi, gti) in zip(x, gt)])
    #         metrics_test = {'psnr-mean': psnr_mean, 'ssim-mean': ssim_mean, 'mse-mean': mse_mean, 're-mean': re_mean, 'ae-mean': ae_mean, 'dr-mean': dr_mean,
    #                         'psnr-var': psnr_var, 'ssim-var': ssim_var, 'mse-var': mse_var, 're-var': re_var, 'ae-var': ae_var, 'dr-var': dr_var}
    #         self.log_dict(metrics_test)
    #         # self.logger.experiment.add_image("reco_std", xstd, global_step=self.current_epoch)
    #
    #         imgs = torch.cat([gt, y, torch.clamp(x, 0, 2)], axis=0)
    #         # imgs = torch.cat([gt, y, x], axis=0)
    #         grid = torchvision.utils.make_grid(imgs, nrow=num_show)
    #         self.logger.experiment.add_image("test-truth-gn-reconstructions", grid, global_step=self.current_epoch,
    #                                          dataformats='CHW')
    #         # grid = grid.cpu().numpy()
    #         # fig, axes = plt.subplots(1,2,figsize=(20,10))
    #         # a0 = axes[0].imshow(grid[0])
    #         # a1 = axes[1].imshow(grid[1])
    #         # fig.colorbar(a0, ax=axes, orientation='vertical', fraction=.1, pad=0.025,
    #         #               ticks=[0, 0.3, 0.6, 0.9, 1.2, 1.5])
    #         # self.logger.experiment.add_figure("test-truth-gn-reconstructions1", fig, global_step=self.current_epoch)

    def configure_optimizers(self):
        """
        Setup the optimizer. Currently, the ADAM optimizer is used.
        Returns
        -------
        optimizer : torch optimizer
            The Pytorch optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.001,
                                     weight_decay=1e-5) #lr=0.001,weight_decay=1e-5

        sched_factor = 0.9  # new_lr = lr * factor
        sched_patience = 2
        sched_tresh = 0.005
        sched_cooldown = 1

        reduce_on_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=sched_factor,
            patience=sched_patience, threshold=sched_tresh,
            min_lr=1e-10, eps=1e-08, cooldown=sched_cooldown,
            verbose=False)

        schedulers = {
            'scheduler': reduce_on_plateu,
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1}

        return [optimizer], [schedulers]


# %
if __name__ == '__main__':
    log_dir = os.path.join(*['..', 'exp', '25db_2'])
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval=None)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir, version=f'bs_8_sf_0.9_cfs_256_epoch_320_clamp2.5')

    trainer_args = {'gpus': [0],
                    'default_root_dir': log_dir,
                    'callbacks': [checkpoint_callback, lr_monitor],
                    'benchmark': False,
                    'fast_dev_run': False,
                    'limit_train_batches': 1.0,  # 训练数据集的使用比例，用来测试和debug。
                    'gradient_clip_val': 1,
                    'logger': tb_logger}
                    # 'weights_summary': None} 1
    pprint(trainer_args)
    # %
    H = W = 128
    downsample_levels = 3   # 不能改这个参数
    img_size = check_img_size(H, W, downsample_levels)
    if img_size[0] == H and img_size[1] == W:
        print(f'img_size ok!!!')
    """
    112img_size ok!!!
    128img_size ok!!!
    144img_size ok!!!
    160img_size ok!!!
    176img_size ok!!!
    192img_size ok!!!
    208img_size ok!!!
    """
    # %
    # model = CINN(in_ch=1,
    #              img_size=(H, W),
    #              num_blocks=2,  # cond   _add_conditioned_section(...num_blocks...)
    #              coupling='affine',  # cond
    #              downsample_levels=downsample_levels,  # cond
    #              cond_conv_channels=[4, 16, 32, 64, 64, 32],  # cond
    #              downsampling='invertible',
    #              # cond xxx 在downsampling='standard'时候，_add_downsample中的subnetUncond的in_ch,out_ch出错(out_ch=0)
    #              cond_fc_size=128,
    #              num_fc=3,
    #              use_act_norm=True,
    #              add_reg=0)
    # print(model)

    # summary(model)

    # trainer = pl.Trainer(max_epochs=300, **trainer_args)
    #
    # num_data_loader_workers = 0
    # batch_size = 16
    # # data_file_list = [os.path.join(r'E:\codePython\eit_cnf\cinn_for_imaging\data',f'eit_circle{i}.npz') for i in range(137,140)]
    # # data_file_list = [os.path.join(r'D:\CNF\eit_cnf\cinn_for_imaging\data', f'eit_circle{i}.npz') for i in
    # #                   range(137, 140)]
    # # data_file_list = [os.path.join(r'F:\eit_cnf\cinn_for_imaging\27000_eit_noise0.005_h00.05', f'eit_circle{i}.npz') for i in
    # #                   range(0, 10)]
    # # # data_file_list = [os.path.join(r'D:\CNF\eit_cnf\cinn_for_imaging\data', f'{i}.npz') for i in
    # # #                   range(0, 6)]
    # # num_train_data_file = 8
    # # num_val_data_file = 1
    # # train_data_file_list = data_file_list[:num_train_data_file]
    # # val_data_file_list = data_file_list[num_train_data_file:num_train_data_file + num_val_data_file]
    # # test_data_file_list = data_file_list[num_train_data_file + num_val_data_file:]
    #
    # data_file_list_1 = [os.path.join(r'/media/fwq/个人数据/WHH/paper/data/50db_2_9000', f'{i}.npz') for i in
    #                     range(0, 10)]
    # data_file_list_2 = [os.path.join(r'/media/fwq/个人数据/WHH/paper/data/50db_3_9000', f'{i}.npz') for i in
    #                     range(0, 10)]
    # data_file_list_3 = [os.path.join(r'/media/fwq/个人数据/WHH/paper/data/50db_4_9000', f'{i}.npz') for i in
    #                     range(0, 10)]
    #
    # train_data_file_list = data_file_list_1[0:8] + data_file_list_2[0:8] + data_file_list_3[0:8]
    # val_data_file_list = data_file_list_1[8:9] + data_file_list_2[8:9] + data_file_list_3[8:9]
    # test_data_file_list = data_file_list_1[9:] + data_file_list_2[9:] + data_file_list_3[9:]
    # dataset = EITDataModule(train_data_file_list=train_data_file_list,
    #                         val_data_file_list=val_data_file_list,
    #                         test_data_file_list=test_data_file_list,
    #                         target_size=H,
    #                         num_data_loader_workers=num_data_loader_workers,
    #                         batch_size=batch_size)
    # trainer.fit(model, datamodule=dataset)
    # trainer.test(model, datamodule=dataset)
    # # test_data = np.load(r'D:\eit_cnf\cinn_for_imaging\27000_eit_no_noise_h00.05_hpc\eit_circle9.npz')
    # # xs_test = test_data['xs']
    # # xs0 = xs_test[0]
    num_data_loader_workers = 0
    batch_size =16
    train_data_file_list = [os.path.join(r'/media/fwq/个人数据/WHH/paper/data/25db_2', f'{i}.npz') for i in
                            range(0, 24)]
    val_data_file_list = [os.path.join(r'/media/fwq/个人数据/WHH/paper/data/25db_2', f'{i}.npz') for i in
                          range(24, 27)]
    test_data_file_list = [os.path.join(r'/media/fwq/个人数据/WHH/paper/data/25db_2', f'{i}.npz') for i in
                           range(27, 28)]

    # train_data_file_list = data_file_list_1[0:8] + data_file_list_2[0:8] + data_file_list_3[0:8]
    # val_data_file_list = data_file_list_1[8:9] + data_file_list_2[8:9] + data_file_list_3[8:9]
    # # test_data_file_list = data_file_list_1[9:] + data_file_list_2[9:] + data_file_list_3[9:]
    # test_data_file_list = data_file_list_2[9:] + data_file_list_1[9:] + data_file_list_3[9:]
    dataset = EITDataModule(train_data_file_list=train_data_file_list,
                            val_data_file_list=val_data_file_list,
                            test_data_file_list=test_data_file_list,
                            target_size=H,
                            num_data_loader_workers=num_data_loader_workers,
                            batch_size=batch_size)

    model = CINN(in_ch=1,
                 img_size=(H, W),
                 num_blocks=2,  # cond   _add_conditioned_section(...num_blocks...)
                 coupling='affine',  # cond
                 downsample_levels=downsample_levels,  # cond
                 cond_conv_channels=[4, 16, 32, 64, 64, 32],  # cond
                 downsampling='invertible',
                 # cond xxx 在downsampling='standard'时候，_add_downsample中的subnetUncond的in_ch,out_ch出错(out_ch=0)
                 cond_fc_size=256,
                 num_fc=3,
                 use_act_norm=True,
                 add_reg=0)
    # 方法一
    # model = CINN.load_from_checkpoint('tmp/pl_example.ckpt')

    # 方法二
    RESUME, TEST =False,True
    if not RESUME:
        trainer = pl.Trainer(max_epochs=320, **trainer_args)
        trainer.fit(model, datamodule=dataset)

    else:
        if not TEST:
            resume_checkpoint_dir = r'../exp/50db_4_perm2/lightning_logs/bs_8_sf_0.9_cfs_512_epoch_300/checkpoints/'  # 补充正确的check_point路径
            checkpoint_path = os.listdir(resume_checkpoint_dir)[0]
            resume_checkpoint_path = resume_checkpoint_dir + checkpoint_path

            model.load_state_dict(torch.load(resume_checkpoint_path)['state_dict'],strict=False)

            trainer = pl.Trainer()
            trainer.fit(model, datamodule=dataset)
        else:
            test_checkpoint_dir = r'../exp/40db_2/lightning_logs/bs_8_sf_0.9_cfs_128_epoch_320_clamp2.5/checkpoints/'
            checkpoint_path = os.listdir(test_checkpoint_dir)[0]
            test_checkpoint_path = test_checkpoint_dir + checkpoint_path

            model.load_state_dict(torch.load(test_checkpoint_path)['state_dict'],strict=False)
            model.eval()


            MMse, PPsnr, SSsim, AAe, RRe, DDr = [], [], [], [], [], []
            for idx, images in enumerate(dataset.test_dataloader()):
                # z = torch.randn((images[1].shape[0], img_size[0] * img_size[1]))
                with torch.no_grad():
                    Mse, Psnr, Ssim, Ae, Re, Dr = [], [], [], [], [], []
                    for _ in range(20):
                        z = torch.randn((images[1].shape[0], img_size[0] * img_size[1]))
                        x_gn = model(z, images[0], True)
                        mse = torch.nn.MSELoss()(x_gn.cpu(), images[1].cpu())
                        re = Re_sigma(x_gn.cpu(), images[1].cpu())
                        ae = torch.nn.L1Loss()(x_gn.cpu(), images[1].cpu())
                        dr = DR(x_gn.cpu(), images[1].cpu())
                        ssim = SSIM(x_gn.cpu().numpy().squeeze(), images[1].cpu().numpy().squeeze())
                        psnr = PSNR(x_gn.cpu(), images[1].cpu())
                        Mse.append(mse)
                        Psnr.append(psnr)
                        Ssim.append(ssim)
                        Re.append(re)
                        Ae.append(ae)
                        Dr.append(dr)
                    # mse_mean = np.mean([((x_gn.cpu() - images[1].cpu()) ** 2).mean()])
                    mse_mean = np.mean(Mse)
                    psnr_mean = np.mean(Psnr)
                    ssim_mean = np.mean(Ssim)
                    re_mean = np.mean(Re)
                    ae_mean = np.mean(Ae)
                    dr_mean = np.mean(Dr)
                MMse.append(mse_mean)
                PPsnr.append(psnr_mean)
                SSsim.append(ssim_mean)
                RRe.append(re_mean)
                AAe.append(ae_mean)
                DDr.append(dr_mean)
            mmse_mean = np.mean(MMse)
            ppsnr_mean = np.mean(PPsnr)
            sssim_mean = np.mean(SSsim)
            rre_mean = np.mean(RRe)
            aae_mean = np.mean(AAe)
            ddr_mean = np.mean(DDr)
            mse_var = np.std(MMse)
            psnr_var = np.std(PPsnr)
            ssim_var = np.std(SSsim)
            re_var = np.std(RRe)
            ar_var = np.std(AAe)
            dr_var = np.std(DDr)


            print()

                    # fig, ax = plt.subplots(1, 2)
                    # ax[0].imshow(x_gn[0].squeeze().cpu().numpy(), vmin=0, vmax=1.5)
                    # ax[1].imshow(images[1][0].detach().clamp(0, 1.5).squeeze().cpu().numpy(), vmin=0, vmax=1.5)
                    # # plt.title(f'PSNR: {psnr:.7f}, MSE:{mse:.7f}  \n SSIM: {ssim:.7f},')
                    # plt.show()



