# plots topographic map of EEG

# File->Default Settings->Project Interpretor - add numpy
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import scipy.signal as sps
import mne
import csv

# Reads the XY electrode locations from file
# Input - file name
# Output - loc structure
def ReadElectrodeLocations(fileName):
    loc = np.empty((0,2), float)
    with open('ADS35_XY.txt', 'r') as f:
        for l in f:
            s = l.strip().split("\t")
            loc = np.append(loc, np.array([[float(s[0]), float(s[1])]]), axis=0)
    return loc


# Reads BIN file
def ReadBIN(fileName):

    fileHandler = open(fileName, "rb")

    szFile = os.path.getsize(fileName)
    szHeader = 0

    for i in range(50):
        line = fileHandler.readline() # Get next line from file
        szHeader += len(line)
        line = line.decode("utf-8") #bytes to string

        if 'HEADER_END' in line: #break if reached end of header
            break;

        # SamplingRate;1000
        if 'SamplingRate' in line:
            s = line.split(';')
            eegFS = int(s[1])

        # Gain;24
        if 'Gain' in line:
            s = line.split(';')
            gain = int(s[1])

        # Channels;40
        if 'Channels' in line:
            s = line.split(';')
            channels = int(s[1])

        # BytesPerSample;168
        if 'BytesPerSample' in line:
            s = line.split(';')
            bytesPerSample = int(s[1])

        # RecordsPerSample;42
        if 'RecordsPerSample' in line:
            s = line.split(';')
            recordsPerSample = int(s[1])


    # get back to start
    fileHandler.seek(0)

    # skip header
    fileHandler.read(szHeader);

    # number of samples EEG
    szEEG = szFile - szHeader
    NsamplesEEG = float(szEEG) / float(bytesPerSample)

    # check integer amount
    if float(szEEG) % float(bytesPerSample) > 0:
         print('Non-integer number of samples !')

    NsamplesEEG = int(NsamplesEEG)

    # load data into matrix (rows channels, columns = samples)
    data = np.zeros( (recordsPerSample, NsamplesEEG) )
    for j in range(NsamplesEEG):
        for i in range(recordsPerSample):
            x = struct.unpack('I', fileHandler.read(4))[0]
            x = x / 8388608 * 4.5 # for simplicity
            '''
            if x > 0:
                x = x / 8388607 * 4.5
            else: 
                x = x / 8388608 * 4.5
            '''
            x = x / gain;
            data[i, j] = x

    # Close file
    fileHandler.close()

    # extract TS,IO,EEG
    eegTS = data[0, :]
    eegIO = data[1, :]
    eegData = data[2:channelsEEG+2, :]

    return eegFS,eegTS,eegIO,eegData


# Variables
channelsEEG = 35 # number of EEG channels
fileNameBIN = 'breath1_eeg.bin'
alphaFrequencyHz = np.array([8,12])

# read electrode locations
loc = ReadElectrodeLocations('ADS35_XY.txt')

# read BIN file
eegFS,eegTS,eegIO,eegData = ReadBIN(fileNameBIN)

# compute spectrum for ech channel
# averag spectrum within alpha band
dataMean = np.zeros( channelsEEG )
for i in range(0, channelsEEG):
    f_welch, S_xx_welch = sps.welch(eegData[i,:], fs=eegFS)
    k = (f_welch > alphaFrequencyHz[0]) & (f_welch < alphaFrequencyHz[1])
    S = S_xx_welch[k]
    mS = np.mean(S)
    dataMean[i] = mS

# z-score normalize output
dataMean = dataMean - np.mean(dataMean)
dataMean = dataMean / np.std(dataMean)

# print(dataMean)

#plot topographical representation
# plot_topomap(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
#                  res=64, axes=None, names=None, show_names=False, mask=None,
#                  mask_params=None, outlines='head',
#                  contours=6, image_interp='bilinear', show=True,
#                  head_pos=None, onselect=None, extrapolate='box'):

'''
methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
'''

mne.viz.plot_topomap(dataMean, loc, vmin=-2, vmax=2, cmap=None, sensors=True,
                 res=64, axes=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head',
                 contours=6, image_interp='bilinear', show=True,
                 head_pos=None, onselect=None, extrapolate='box')



