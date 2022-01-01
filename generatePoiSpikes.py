import numpy as np

def generatePoiSpikes(r, dt, totalSize):
    spike_train = np.random.poisson(r, totalSize/dt)
    return spike_train
