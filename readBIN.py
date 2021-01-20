# BIN converter

# File->Default Settings->Project Interpretor - add numpy

import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import scipy.signal as sps

filterDC = 1
eegFS = 1000
fc = .5/eegFS # -3dB at this frequency
k = 0.5
alpha = (1-k*np.cos(2*np.pi*fc)-np.sqrt(2*k*(1-np.cos(2*np.pi*fc))-np.square(k)*np.square(np.sin(2*np.pi*fc))))/(1-k)
bDC = 1 - alpha
aDC = [1, -1*alpha];

channelsEEG = 35;

fileName = '../2019_01_07_Meditation/breath1_eeg.bin'

# Open file
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

# print(dataInt.shape)

# extract TS,IO,EEG
eegTS = data[0, :]
eegIO = data[1, :]
eegData = data[2:channelsEEG+2, :]

# apply DC filter
if filterDC == 1:
    for i in range(0, channelsEEG):
        xi = eegData[i, :]
        xiDC = sps.filtfilt(bDC,aDC,xi) #filter
        xi = xi - xiDC # remove DC
        eegData[i, :] = xi

# plot data
for i in range(0, channelsEEG):
    xi = eegData[i, 10*eegFS:12*eegFS]
    # xi = xi-np.mean(xi) # remove mean
    xi = xi-i*0.00001   # display channel with offset
    plt.plot(xi)

plt.show()


