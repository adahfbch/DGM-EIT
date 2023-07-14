"""
Baseline cINN model from the Master thesis of Alexander Denker for the 
LoDoPaB-CT dataset.
"""

from operator import mod
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import torchvision

import numpy as np
import pytorch_lightning as pl
from dival.util.torch_utility import TorchRayTrafoParallel2DAdjointModule
from odl.tomo.analytic import fbp_filter_op
from odl.contrib.torch import OperatorModule

from util.torch_losses import CINNNLLLoss
from reconstructors.networks.layers import NICECouplingBlock, InvertibleDownsampling, Fixed1x1ConvOrthogonal, Split
from reconstructors.networks.cond_net import SimpleCondNetFBP, AvgPoolCondNetFBP, ResNetCondNet
from reconstructors.networks.unet import UNet

class PREINN(nn.Module):
    """
    PyTorch cINN architecture for low-dose CT reconstruction.
    
    Attributes
    ----------
    cinn : torch module list
        Building blocks of the conditional network.
    cond_net : torch module list
        Building blocks of the conditional network.

    Methods
    -------
    forward(c)
        Compute the forward pass.
        
    """
    def __init__(self,in_ch:int,img_size,
                 sample_distribution:str = 'normal',
                 optimizer_args: dict = {'lr': 0.001, 'weight_decay': 1e-5},
                 downsample_levels: int = 3,
                 num_fc:int = 4,
                 clamping: float = 2.5,
                 downsampling: str ='standard',
                 coupling: str ='affine',
                 use_fc_block: bool = True,
                 use_act_norm: bool = False,
                 num_blocks: int = 6,
                 permutation: str = '1x1',
                 train_noise = (0.,0.),
                 **kwargs):
        """
        LowDoseCINN constructor.

        Parameters
        ----------
        sample_distribution: str, optional
            Distribution Z of the random samples z.
            The default is 'normal'
            and frequency_scaling=1.
        optimizer_args : dict, optional
            Arguments for the optimizer.
            The defaults are lr=0.001 and weight_decay=1e-5 for the ADAM
            optimizer.
        downsample_levels : int, optional
            Number of 1/2 downsampling steps in the network. This option is 
            currently not in active use for the structure of the network!
            The default is 5.
        num_fc : int, optional
            Number of fully connected blocks at the end of the cINN network.
            The default is 4.
        clamping : float, optional
            The default is 1.5.
        downsampling : str, optional  文章Figure 1, Section2.4.2
            Type of the downsampling layers. Options are:
                'reshape': Only reshape downsampling
                'haar': Only Haar downsampling
                'invertible': Only learned invertible downsampling
            The default is 'standard'.
        use_fc_block: bool
            Wether to use a last fully connected block
        permutation: str = '1x1',
            Type of permutation to use after coupling blocks
                '1x1' : Fixed1x1 Convolution 
                'PermuteRandom: Random channel wise permutation
        Returns
        -------
        None.

        """
        super().__init__()
        
        # shorten some of the names or store values that can't be placed in
        # a .yml file
        self.permutation = permutation
        self.num_blocks = num_blocks
        self.clamping = clamping
        self.num_fc = num_fc
        self.use_act_norm = use_act_norm
        self.downsampling = downsampling
        self.use_fc_block = use_fc_block
        self.downsample_levels = downsample_levels
        self.coupling = coupling
        self.in_ch = in_ch
        self.img_size = img_size
        self.data_range = [1e-05,2.5]

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

        ### build the network & add conditioning and splits ###

        ## 1) Input region + first downsampling (1 x 1 x 1 -> 4 x 1/2 x 1/2)
        nodes = [Ff.InputNode(self.in_ch, self.img_size[0], self.img_size[1],name='inp')]
        
        _add_downsample(nodes, self.downsampling, coupling=self.coupling, use_act_norm=self.use_act_norm)
        
        for downsample_step in range(self.downsample_levels -1 ):
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
                            {'section_sizes': [128], 'dim' : 0, 'n_sections': None},
                            name="split_fc"))
            split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                            name='flatten_split_fc'))
            ## 4) Random Permute -> Fully Connected Cond
            for k in range(self.num_fc):
                nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed':k},
                                    name='Permute_{}'.format(k)))

                if self.coupling == 'affine':
                    nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock, 
                                    {'subnet_constructor':subnet_fc,
                                    'clamp':self.clamping},
                                    name='GlowBlock_fc_{}_{}'.format(
                                        self.downsample_levels + 1, k)))
                else: 
                    nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,  
                                    {'subnet_constructor':subnet_fc},
                                    name='NICEBlock_fc_{}_{}'.format(
                                        self.downsample_levels + 1, k)))


        ## 5) concat all split notes and network output
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}, name='concat_splits'))
  
        nodes.append(Ff.OutputNode(nodes[-1], name='out'))
        
        return Ff.GraphINN(nodes + split_nodes,verbose=False)
    
    def init_params(self):
        """
        Initialize the parameters of the model.

        Returns
        -------
        None.

        """
        # approx xavier
        for key, param in self.cinn.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.02 * torch.randn(param.data.shape)
                # last convolution in the coeff func
                if len(split) > 3 and split[3][-1] == '4': 
                    param.data.fill_(0.)

    def forward(self, cinn_input, rev:bool = True,
                cut_ouput:bool = True):
        """
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
        # direction Z -> X
        if rev:
            x, _ = self.cinn(cinn_input, rev=rev)
            if cut_ouput:
                return x[:,:,:self.op.domain.shape[0],:self.op.domain.shape[1]]
            else:
                return x
        # direction X -> Z
        else:
            z, log_jac = self.cinn(cinn_input, rev=rev)
            return z, log_jac

def _add_downsample(nodes, downsample, coupling, clamping=2, use_act_norm=True):
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
    return nn.Sequential(
                nn.Conv2d(in_ch, 2*in_ch, 1),
                nn.LeakyReLU(),
                nn.Conv2d(2*in_ch, 2*in_ch, 1),
                nn.LeakyReLU(),
                nn.Conv2d(2*in_ch, out_ch, 1))

if __name__ == '__main__':
    c = PREINN(in_ch=1,img_size=(384,384))
    c(torch.randn(5,1,384,384).cuda(), rev=True)