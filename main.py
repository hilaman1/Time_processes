import numpy as np
import matplotlib.pyplot as plt

import params
import spike_train_utils

# Part 1 - Poisson Process
poisson_spikes = spike_train_utils.generate_poisson_spikes(params.r, params.dt, params.total_size)
fano_factor = spike_train_utils.calc_FF(poisson_spikes)
cv = spike_train_utils.calc_CV(poisson_spikes)

# Part 2 - Refractory Period and correlation

# 2.2
r0 = np.full(int(params.total_size / params.dt), params.r0)
spike_train_utils.generate_poisson_spikes_with_refractory_period(r0, params.dt, params.total_size)

# 2.4
bursty_firing_rate = spike_train_utils.generate_bursty_firing_rate(int(params.total_size / params.dt))
bursty_spike_train = spike_train_utils.generate_poisson_spikes_with_refractory_period(bursty_firing_rate, params.dt,
                                                                                      params.total_size)

# 2.5
acorr = np.correlate(bursty_spike_train, bursty_spike_train, mode='full')
plt.title("Autocorrelation")
plt.plot(acorr)
plt.grid(True)
plt.show()