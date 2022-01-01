import numpy as np


def calcRate(spikeTrain, window):
    """ receive a spike train and length of the window """

    # if window == 0:
    #     rateOfFire = np.mean(spikeTrain)

    w = np.full(window, 1 / window) # create the window to convolve with
    rateOfFire = np.convolve(spikeTrain, w, mode = 'valid')
    return rateOfFire
