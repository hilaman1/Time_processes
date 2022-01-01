import numpy as np

def calcFF(spikeTrain):
    FF = np.var(spikeTrain)/np.mean(spikeTrain)
    # FF should be 1 if it's Poisson
    return FF