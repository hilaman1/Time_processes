import params
import spike_train_utils

poisson_spikes = spike_train_utils.generate_homogeneous_poisson_spikes(params.r, params.dt, params.total_size)
fano_factor = spike_train_utils.calc_FF(poisson_spikes)
cv = spike_train_utils.calc_CV(poisson_spikes)

# Refractory Period and correlation
# 2.2
# spike_train_utils.generate_poisson_spikes_with_refractory_period(params.r0, params.dt, params.total_size)

# 2.4
bursty_firing_rate = spike_train_utils.generate_bursty_firing_rate(int(params.total_size / params.dt))