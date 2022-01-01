import numpy as np


def generatePoiSpikes(r, dt, totalSize):
    if isinstance(r, np.ndarray):
        spike_train = np.Generator.poisson(r, totalSize / dt)
    else:
        spike_train = np.random.poisson(r, totalSize / dt)
    return spike_train
