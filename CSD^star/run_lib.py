"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
import cs
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint, Re_sigma, DR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import random

FLAGS = flags.FLAGS

# #
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(config, workdir):
    """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    if config.training.resume:
        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_EIT_dataset(config)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        try:
            batch = next(train_iter)[1].to(config.device).float()
        except StopIteration:
            train_iter = iter(train_ds)
            batch = next(train_iter)[1].to(config.device).float()

        batch = scaler(batch)
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            try:
                eval_batch = next(eval_iter)[1].to(config.device).float()
            except StopIteration:
                eval_iter = iter(eval_ds)
                eval_batch = next(eval_iter)[1].to(config.device).float()
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            # save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                writer.add_image('val_sample', image_grid, step)

def evaluate(config,
             workdir,
             eval_folder="eval"):
    """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
    # Create directory to eval_folder
    # eval_dir = os.path.join(workdir, eval_folder)
    # tf.io.gfile.makedirs(eval_dir)

    # seed
    setup_seed(config.seed)

    # Build data pipeline
    _, _, test_ds = datasets.get_EIT_dataset(config)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean,
                                       continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (1,
                          config.data.num_channels,
                          config.data.image_size, config.data.image_size
                          )
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    waiting_message_printed = False
    logging.info("begin checkpoint: %d" % (config.eval.best_ckpt))
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(config.eval.best_ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
        if not waiting_message_printed:
            logging.warning("Waiting for the arrival of checkpoint_%d" % (config.eval.best_ckpt,))
            waiting_message_printed = True
        time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{config.eval.best_ckpt}.pth')
    try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
        time.sleep(60)
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(120)
            state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    # if config.eval.enable_loss:
    all_losses, Psnr, Ssim, Mse, Re, Ae, DDR = [], [], [], [], [], [], []
    eval_iter = iter(test_ds)  # pytype: disable=wrong-arg-types
    for i, batch in enumerate(eval_iter):
        eval_batch = batch[1].to(config.device).float()
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if i == -1:
            logging.info("Finished %dth step loss evaluation,mean_loss: %d" % (i + 1, torch.mean(all_losses)))

        xs_inv = scaler(batch[0].to(config.device).float())
        y = batch[2].cpu().numpy()
        sample, n = sampling_fn(score_model, xs_inv)

        ground_truth = eval_batch
        from dival.measure import PSNR, SSIM
        psnr = PSNR(ground_truth.squeeze().cpu().detach().numpy(),
                    sample.clamp(1e-05, 2.5).squeeze().cpu().detach().numpy())
        ssim = SSIM(ground_truth.squeeze().cpu().detach().numpy(),
                    sample.clamp(1e-05, 2.5).squeeze().cpu().detach().numpy())

        mse = torch.nn.MSELoss()(ground_truth.squeeze().cpu().detach(),
                                 sample.clamp(1e-05, 2.5).squeeze().cpu().detach())
        re = Re_sigma(ground_truth.squeeze().cpu().detach(),
                                 sample.clamp(1e-05, 2.5).squeeze().cpu().detach())
        dr = DR(ground_truth.squeeze().cpu().detach(),
                                 sample.clamp(1e-05, 2.5).squeeze().cpu().detach())
        ae = torch.nn.L1Loss()(ground_truth.squeeze().cpu().detach(),
                                 sample.clamp(1e-05, 2.5).squeeze().cpu().detach())

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(ground_truth.squeeze().cpu().numpy(), vmin=0, vmax=2)
        # ax[1].imshow(sample.detach().clamp(1e-05, 2).squeeze().cpu().numpy(), vmin=0, vmax=2)
        # plt.title(f'PSNR: {psnr:.7f}, MSE:{mse:.7f}  \n SSIM: {ssim:.7f},')
        # plt.show()
        # writer.add_figure('test',fig,i)

        Psnr.append(psnr)
        Ssim.append(ssim)
        Mse.append(mse)
        Re.append(re)
        Ae.append(ae)
        DDR.append(dr)
        logging.info(f'PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, MSE:{mse:.4f}')
        # print(f'PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, MSE:{mse:.4f}')
        # writer.add_scalar("PSNR", psnr, i)
        # writer.add_scalar("SSIM", ssim, i)
        # writer.add_scalar("MSE", mse, i)
    mse_mean = np.mean(Mse)
    psnr_mean = np.mean(Psnr)
    ssim_mean = np.mean(Ssim)
    re_mean = np.mean(Re)
    ae_mean = np.mean(Ae)
    dr_mean = np.mean(DDR)
    mse_std = np.std(Mse)
    psnr_std = np.std(Psnr)
    ssim_std = np.std(Ssim)
    re_std = np.std(Re)
    ae_std = np.std(Ae)
    dr_std = np.std(DDR)
    print(0)
