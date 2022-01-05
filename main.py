import numpy as np
from generatePoiSpikes import generatePoiSpikes
import matplotlib.pyplot as plt

# Question 2.1
spike_train = generatePoiSpikes(30, 0.001, 60*1000)
count, bins, ignored = plt.hist(spike_train, 20, density=True)
plt.xlabel('Time (ms)', fontsize=16)
plt.ylabel('Spike', fontsize=16)
plt.show()

# Question 2.2
firing_rate = [30] + [0 for i in range(1, 60000)]
spike_train = [np.random.poisson(lam=firing_rate[0], size=None)]+[0 for i in range(1, 60000)]
t_spike = 0
for t in range(1, 60*1000):
    firing_rate[t] = firing_rate[0]*(t - t_spike)/5
    spike_train = np.random.poisson(lam=t-1, size=None)












