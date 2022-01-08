import random
import numpy as np
import params
import matplotlib.pyplot as plt


def generatePoiSpikes(r, dt, totalSize):
    return 1 * (np.random.rand(int(totalSize / dt)) < r * dt)


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


def generate_poisson_spikes_with_refractory_period(r0, dt, total_size):
    spikeTrain = np.zeros(int(total_size / dt))
    r = r0
    last_response = - params.refractory_period
    for i in range(len(spikeTrain)):
        r = min(r0, r0 * dt * (i - last_response) / params.refractory_period)
        if np.random.rand(int(total_size / dt)) < r * dt:
            spikeTrain[i] = 1
            last_response = i
    return spikeTrain


def generate_bursty_firing_rate(n):
    r = np.zeros(n)
    s = random.randint(0, int(n / 2))
    e = random.randint(s, n - 1)
    r[s:e] = params.high_firing_rate
    return r


def statsFunctions(spikeTrain):
    tau = np.diff(np.where(spikeTrain == 1))
    [hist, bins] = np.histogram(tau * params.dt, np.linspace(0.5 * params.dt, np.max(tau) * dt + dt / 2, int(np.max(tau) + 1)))
    plt.bar(bins[:-1], hist, align='edge')
    pdf_tau = hist / float(len(tau[0]))
    cdf_tau = np.cumsum(pdf_tau)
    survival = 1 - cdf_tau
    plt.plot(bins[1:] + (bins[1] - bins[0]) / 2, survival)
    hazard = pdf_tau / survival
    plt.plot(bins[0:-3] + (bins[1] - bins[0]) / 2, hazard[:-2])

