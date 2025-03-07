def get_projection_sampler(config, sde, model, shape, predictor, corrector,
                           inverse_scaler, n_steps=1,
                           probability_flow=False, continuous=True,
                           denoise=True, eps=1e-5):
  if config.sampling.task == 'mri':
    to_space = lambda x: get_kspace(x, (1, 2))
    from_space = lambda x: kspace_to_image(x, (1, 2)).real

  elif config.sampling.task in ('ct', 'mar', 'sparse_mar'):
    to_space = lambda x: fft_radon_to_kspace(x[..., 0], config.sampling.expansion)[..., None]
    from_space = lambda x: fft_radon_to_image(x[..., 0], config.data.image_size)[..., None]

  else:
    raise ValueError(f'Task {config.sampling.task} not recognized.')

  def get_inpaint_update_fn(update_fn):
    def inpaint_update_fn(rng, state, x, t, mask, known, coeff):
      x_space = to_space(x)

      mean, std = sde.marginal_prob(known, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      noise_space = to_space(noise)
      noisy_known = mean + batch_mul(std, noise_space)

      x_space = merge_known_with_mask(config, x_space, noisy_known, mask, coeff)
      x = from_space(x_space)

      rng, step_rng = jax.random.split(rng)
      x, x_mean = update_fn(step_rng, state, x, t)

      return x

    return inpaint_update_fn

  def projection_sampler(rng, state, img, coeff, snr):
    # Initial sample
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape)

    mask = get_masks(config, img)
    known = get_known(config, img)

    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            model=model,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            model=model,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    cs_predictor_update_fn = get_inpaint_update_fn(predictor_update_fn)
    cs_corrector_update_fn = get_inpaint_update_fn(corrector_update_fn)

    timesteps = jnp.linspace(sde.T, eps, sde.N)

    def loop_body(carry, i):
      rng, x = carry
      t = timesteps[i]
      vec_t = jnp.ones(shape[0]) * t
      rng, step_rng = random.split(rng)
      x = cs_corrector_update_fn(step_rng, state, x, vec_t, mask, known, coeff)
      rng, step_rng = random.split(rng)
      x = cs_predictor_update_fn(step_rng, state, x, vec_t, mask, known, coeff)
      output = x
      return (rng, x), output

    _, all_samples = jax.lax.scan(loop_body, (rng, x), jnp.arange(0, sde.N), length=sde.N)

    output = all_samples[-1]
    # output = all_samples
    if denoise:
      t_eps = jnp.full((output.shape[0],), eps)
      k, std = sde.marginal_prob(jnp.ones_like(output), t_eps)
      score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state,
                                     train=False, continuous=continuous, return_state=False)
      score = score_fn(output, t_eps)
      output = output / k + batch_mul(std ** 2, score / k)
      output_space = to_space(output)
      output_space = merge_known_with_mask(config, output_space, known, mask, 1.)
      output = from_space(output_space)

    return inverse_scaler(output)

  return jax.pmap(projection_sampler, axis_name='batch', in_axes=(0, 0, 0, None, None))
