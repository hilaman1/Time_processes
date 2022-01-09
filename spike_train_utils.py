import random
import numpy as np
import params
import matplotlib.pyplot as plt


def generatePoiSpikes(r, dt, totalSize):
    return 1 * (np.random.rand(int(totalSize / dt)) < r * dt)


def generate_poisson_spikes(r, dt, total_size):
    spike_train = np.zeros(int(total_size / dt))
    if type(r) == int or float:
        spike_train[np.random.rand(int(total_size / dt)) < r * dt] = 1
    else:
        # assuming len(r) = len(spike_train) = int(total_size / dt)
        for i in range(len(spike_train)):
            if random.random() < r[i] * dt:
                spike_train[i] = 1
    return spike_train


# Calculate the FF of the given spikeTrain
def calcFF(spikeTrain):
    return np.var(spikeTrain) / np.mean(spikeTrain)


# Calculate the CV of the given spikeTrain
def calcCV(spikeTrain):
    return np.std(spikeTrain) / np.mean(spikeTrain)


# Calculate the firing rate of the given spikeTrain
def calcRate(spikeTrain, window):
    if window == 0:
        return np.mean(spikeTrain)
    w = np.full(window, 1 / window)  # create the window to convolve with
    rate = np.convolve(spikeTrain, w, mode='valid')
    plt.figure()
    plt.plot(np.arange(len(spikeTrain)) * params.dt, rate)
    return rate


def generate_poisson_spikes_with_refractory_period_v2(r0, dt, total_size):
    spikeTrain = np.zeros(int(total_size / dt))
    last_response = - params.refractory_period
    for i in range(len(spikeTrain)):
        r = min(r0[0], r0[0] * dt * (i - last_response) / params.refractory_period)
        if random.random() < r * dt:
            spikeTrain[i] = 1
            last_response = i
    return spikeTrain


def generate_poisson_spikes_with_refractory_period(r0, dt, total_size):
    spike_train = np.zeros(int(total_size / dt))
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

def statsFunctions(spikeTrain):
    tau = np.diff(np.where(spikeTrain == 1))
    [hist, bins] = np.histogram(tau * params.dt, np.linspace(0.5 * params.dt, np.max(tau) * params.dt + params.dt / 2, int(np.max(tau) + 1)))
    plt.bar(bins[:-1], hist, align='edge')
    pdf_tau = hist / float(len(tau[0]))
    cdf_tau = np.cumsum(pdf_tau)
    survival = 1 - cdf_tau
    plt.plot(bins[1:] + (bins[1] - bins[0]) / 2, survival)
    hazard = pdf_tau / survival
    plt.plot(bins[0:-3] + (bins[1] - bins[0]) / 2, hazard[:-2])

