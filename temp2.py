def get_pc_fouriercs_RI_coil_SENSE(sde, predictor, corrector, inverse_scaler, snr,
                                   n_steps=1, lamb_schedule=None, probability_flow=False, continuous=False,
                                   denoise=True, eps=1e-5, sens=None, mask=None, m_steps=10,
                                   save_progress=False, save_root=None):
  '''Every once in a while during separate coil reconstruction,
  apply SENSE data consistency and incorporate information.
  (Args)
    (sens): sensitivity maps
    (m_steps): frequency in which SENSE operation is incorporated
  '''
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  # functions to impose data fidelity 1/2\|Ax - y\|^2
  def data_fidelity(mask, x, x_mean, y):
      x = ifft2_m(fft2_m(x) * (1. - mask) + y)
      x_mean = ifft2_m(fft2_m(x_mean) * (1. - mask) + y)
      return x, x_mean

  def A(x, sens=sens, mask=mask):
      return mask * fft2_m(sens * x)

  def A_H(x, sens=sens, mask=mask):  # Hermitian transpose
      return torch.sum(torch.conj(sens) * ifft2_m(x * mask), dim=1).unsqueeze(dim=1)

  def kaczmarz(x, x_mean, y, lamb=1.0):
      x = x + lamb * A_H(y - A(x))
      x_mean = x_mean + lamb * A_H(y - A(x_mean))
      return x, x_mean

  def get_coil_update_fn(update_fn):
    def fouriercs_update_fn(model, data, x, t, y=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        # split real / imag part
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        # perform update step with real / imag part seperately
        x_real, x_real_mean = update_fn(x_real, vec_t, model=model)
        x_imag, x_imag_mean = update_fn(x_imag, vec_t, model=model)

        # merge real / imag values to form complex image
        x = x_real + 1j * x_imag
        x_mean = x_real_mean + 1j * x_imag_mean

        # coil mask
        mask_c = mask[0, 0, :, :].squeeze()
        x, x_mean = data_fidelity(mask_c, x, x_mean, y)
        return x, x_mean

    return fouriercs_update_fn

  predictor_coil_update_fn = get_coil_update_fn(predictor_update_fn)
  corrector_coil_update_fn = get_coil_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, y=None):
    with torch.no_grad():
      # Initial sample: [1, 15, 320, 320] (dtype: torch.complex64)
      x_r = sde.prior_sampling(data.shape).to(data.device)
      x_i = sde.prior_sampling(data.shape).to(data.device)
      x = torch.complex(x_r, x_i)
      x_mean = x.clone().detach()

      timesteps = torch.linspace(sde.T, eps, sde.N)

      # number of iterations of PC sampler
      for i in tqdm(range(sde.N)):
        # coil x_c update
        for c in range(15):
          t = timesteps[i]

          # slicing the dimension with c:c+1 ("one-element slice") preserves dimension
          x_c = x[:, c:c+1, :, :]
          y_c = y[:, c:c+1, :, :]
          x_c, x_c_mean = predictor_coil_update_fn(model, data, x_c, t, y=y_c)
          x_c, x_c_mean = corrector_coil_update_fn(model, data, x_c, t, y=y_c)

          # Assign coil dates to the global x, x_mean
          x[:, c, :, :] = x_c
          x_mean[:, c, :, :] = x_c_mean

        # global x update
        if i % m_steps == 0:
          lamb = lamb_schedule.get_current_lambda(i)
          x, x_mean = kaczmarz(x, x_mean, y, lamb=lamb)
        if save_progress:
          if i % 100 == 0:
            for c in range(15):
              x_c = clear(x[:, c:c+1, :, :])
              plt.imsave(save_root / 'recon' / f'coil{c}' / f'after{i}.png', np.abs(x_c), cmap='gray')
            x_rss = clear(root_sum_of_squares(torch.abs(x), dim=1).squeeze())
            plt.imsave(save_root / 'recon' / f'after{i}.png', x_rss, cmap='gray')

      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs
