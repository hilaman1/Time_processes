import numpy as np


def generatePoiSpikes(r, dt, totalSize):
    if isinstance(r, np.ndarray):
        spike_train = np.Generator.poisson(r, int(totalSize / dt))
    else:
        spike_train = np.random.poisson(r, int(totalSize / dt))
    return spike_train
