import numpy as np
import params
import spike_train_utils
import matplotlib.pyplot as plt

poisson_spikes = spike_train_utils.generatePoiSpikes(params.r, params.dt, params.total_size)
fano_factor = spike_train_utils.calcFF(poisson_spikes)
cv = spike_train_utils.calcCV(poisson_spikes)

# Refractory Period and correlation
# 2.1
spikeTrain = spike_train_utils.generatePoiSpikes(30, params.dt, 60)
fig, ax = plt.subplots()
ax.eventplot(np.where(spikeTrain == 1))
plt.show()

# 2.2
r0 = np.full(int(params.total_size / params.dt), params.r0)
spike_train_utils.generate_poisson_spikes_with_refractory_period(r0, params.dt, params.total_size)

# 2.3
spike_train_utils.statsFunctions(poisson_spikes)

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