import numpy as np
import torch
import tensorflow as tf
import os
import logging

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


# def Re_sigma(x, reco_eit):
#     Re_up = torch.nn.L1Loss()(x, reco_eit)
#     t = torch.zeros(x.shape).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     Re_down = torch.nn.L1Loss()(x, t)
#     Re_sigma = Re_up / Re_down
#     return Re_sigma
#
# def DR(x, reco_eit):
#     DR_up = torch.argmax(reco_eit) - torch.argmin(reco_eit)
#     DR_down = torch.argmax(x) - torch.argmin(x)
#     DR = DR_up / DR_down
#     return DR

def restore_checkpoint(ckpt_dir, state, device):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
