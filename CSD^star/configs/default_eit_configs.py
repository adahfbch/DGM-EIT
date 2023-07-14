import ml_collections
import torch

def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 64
  training.n_iters = 500001
  training.snapshot_freq = 400
  training.log_freq = 50
  training.eval_freq = 400
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 3000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.n_jitted_steps = 1
  training.reduce_mean = False
  training.resume = True


  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'ald'
  sampling.n_steps_each = 1
  sampling.noise_removal =True
  sampling.probability_flow = False
  sampling.snr = 0.206
  sampling.denoise_override = True
  sampling.sample_size = 16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  # evaluate.ckpt_id = 101
  evaluate.best_ckpt = 44500
  evaluate.batch_size = 128
  evaluate.enable_sampling = True
  evaluate.num_samples = 2700
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.random_flip = False
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378.
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config