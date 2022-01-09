import numpy as np
import params
import spike_train_utils
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

# Part 1 - Poisson Process
poisson_spikes = spike_train_utils.generatePoiSpikes(params.r, params.dt, params.total_size)
fano_factor = spike_train_utils.calcFF(poisson_spikes)
cv = spike_train_utils.calcCV(poisson_spikes)

# Part 2 - Refractory Period and correlation

# 2.1
spikeTrain = spike_train_utils.generatePoiSpikes(params.r0, params.dt, params.total_size)
tau = np.diff(np.where(spikeTrain == 1))
fig, ax = plt.subplots()
ax.eventplot(tau)
ax.set_title("Spikes in spike train")
fig.show()

# 2.2
r0 = np.full(int(params.total_size / params.dt), params.r0)
spike_train_utils.generate_poisson_spikes_with_refractory_period_v2(r0, params.dt, params.total_size)

# 2.3
spike_train_utils.statsFunctions(poisson_spikes)

# 2.4
bursty_firing_rate = spike_train_utils.generate_bursty_firing_rate(int(params.total_size / params.dt))
bursty_spike_train = spike_train_utils.generate_poisson_spikes_with_refractory_period_v2(bursty_firing_rate, params.dt,
                                                                                         params.total_size)

# 2.5
acorr = np.correlate(bursty_spike_train, bursty_spike_train, mode='full')
fig2, ax2 = plt.subplots()
ax2.set_title("Autocorrelation")
ax2.plot(acorr)
ax2.grid(True)
fig2.show()

# 2.6
another_bursty_spike_train = spike_train_utils.generate_poisson_spikes_with_refractory_period_v2(bursty_firing_rate,
                                                                                                 params.dt,
                                                                                                 params.total_size)
fig3, ax3 = plt.subplots()

xcorr = np.correlate(bursty_spike_train, another_bursty_spike_train, "full")
ax3.set_title("cross-correlation")
ax3.plot(xcorr)

shifted_bursty_spike_train = shift(another_bursty_spike_train, params.shift, cval=0)
shifted_xcorr = np.correlate(bursty_spike_train, shifted_bursty_spike_train, "full")
ax3.set_title("cross-correlation")
ax3.plot(shifted_xcorr)

fig3.show()
