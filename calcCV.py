import numpy as np


def calcCV(spike_train):
    # spikeTrain = (spikeTrain > 0)*1
    tau = np.diff(np.where(spike_train == 1))
    CV = np.std(tau) / np.mean(tau)
    # 1/np.mean(tau) should be 1/r if it's poisson
    return CV
