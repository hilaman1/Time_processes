import numpy as np
import random

import params


def generate_poisson_spikes(r, dt, total_size):
    spike_train = np.zeros(int(total_size / dt))
    np.random.seed(2)
    if type(r) == int or float:
        spike_train[np.random.rand(int(total_size / dt)) < r * dt] = 1
    else:
        # assuming len(r) = len(spike_train) = int(total_size / dt)
        for i in range(len(spike_train)):
            if random.random() < r[i] * dt:
                spike_train[i] = 1
    return spike_train


# Calculate the FF of the given spikeTrain
def calc_FF(spike_train):
    var = np.var(spike_train)
    mean = np.mean(spike_train)
    return var / mean


# Calculate the CV of the given spikeTrain
def calc_CV(spike_train):
    return np.std(spike_train) / np.mean(spike_train)


# Calculate the firing rate of the given spikeTrain
def calc_rate(spike_train, window):
    if window == 0 or window > params.total_size:
        window = params.total_size
    window_bin = int(window / params.dt)

    return (spike_train[:window_bin].sum() / window_bin) / params.dt


# this was my solution:

# if window == 0:
#     rateOfFire = np.mean(spikeTrain)
# w = np.full(window, 1 / window) # create the window to convolve with
#     rateOfFire = np.convolve(spikeTrain, w, mode = 'valid')
#     return rateOfFire

def generate_poisson_spikes_with_refractory_period(r0, dt, total_size):
    spike_train = np.zeros(int(total_size / dt))
    np.random.seed(42)
    was_spike = False
    r = r0[0]
    for i in range(len(spike_train)):
        if was_spike:
            if last_response == i - 1:
                r = 0
            else:
                r = min(r0[i], r0[i] * (i - last_response) / 5)

        if random.random() < r * dt:
            spike_train[i] = 1
            last_response = i
            was_spike = True
    return spike_train


def generate_bursty_firing_rate(n):
    r = np.full(n, params.r0)
    s = random.randint(0, int(n / 2))
    e = random.randint(s, n - 1)
    r[s:e] = params.high_firing_rate
    return r
