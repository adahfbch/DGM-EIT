
from configs.default_eit_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.batch_size = 64
  training.n_iters = 500001
  training.snapshot_sampling = True
  training.sde = 'vesde'
  training.continuous = True

  # eval
  evaluate = config.eval
  evaluate.num_samples = 50
  evaluate.best_ckpt = 49500
  evaluate.batch_size = 32

  # sampling
  sampling = config.sampling
  sampling.method = 'csds'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'ald'
  sampling.snr = 0.056


  # data
  data = config.data
  data.dataset = 'EIT'
  data.image_size = 128
  data.num_channels = 1
  data.centered = False
  data.random_flip = False
  data.uniform_dequantization = False
  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.sigma_max = 128. 
  model.num_scales = 1000
  model.ema_rate = 0.999
  model.sigma_min = 0.01
  model.beta_min = 0.1
  model.beta_max = 20.
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  # model.nf = 128
  # model.ch_mult = (1, 2, 2, 2)
  # model.num_res_blocks = 4
  # model.attn_resolutions = (16,)
  model.nf = 32
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (20,) 
  model.dropout = 0.
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  # optim
  optim = config.optim
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-3 #2e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
