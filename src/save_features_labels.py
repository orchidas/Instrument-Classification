import numpy as np 
from essentia.standard import *
import matplotlib.pyplot as plt 
from scipy.io.wavfile import read, write
import os


trainPath = '../data/instruments/train/'
testPath = '../data/instruments/test/'
instruments = ['bassoon','violin','saxphone','clarinet']

###########################################################################################################

def extract_features(filepath, mel_order = 15, lpc_order = 15, feature = 'both', plot_mel=False):
    # we start by instantiating the audio loader:
    loader = MonoLoader(filename=filepath)
    # and then we actually perform the loading:
    audio = loader()

    frameSize = 1024
    hopSize = 512
    spectrum = Spectrum(size = frameSize)  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = MFCC(numberCoefficients = mel_order, inputSize = frameSize/2 + 1)
    lpc = LPC(order = lpc_order, type='warped')
    (fs,audio) = read(filepath)
    audio = essentia.array(audio)
    w = Windowing(type = 'hann')
    logNorm = UnaryOperator(type='log')
    mfccs = []
    melbands = []
    melbands_log = []
    wlpcs = []
   

    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        # frame = essentia.array(audio[fstart:fstart+frameSize])
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        melbands_log.append(logNorm(mfcc_bands))
        wlpc = lpc(frame)[0]
        #the first LPC coefficient is always 1
        wlpcs.append(wlpc[1:])

    if plot_mel:
    	melbands_log = essentia.array(melbands_log).T
    	plot_mel_spectrum(melbands_log)
    	

    #do we want to use warped lpcs, mfccs or both?
    if feature is 'wlpc':
    	#first column - warped LPCs 0- select median over all frames
    	features = normalize_features(np.median(wlpcs,axis = 0))
    elif feature is 'mfcc':
    	#second column - mfccs - select median over all frames
    	features = normalize_features(np.median(mfccs, axis = 0))
    else:
    	features = np.zeros([lpc_order+mel_order])
        features[:lpc_order] = normalize_features(np.median(wlpcs,axis = 0))
        features[lpc_order:] = normalize_features(np.median(mfccs, axis = 0))

    return features



def plot_mel_spectrum(melbands_log):
    plt.imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
    plt.title("Log-normalized mel band spectral energies in frames")
    plt.ylabel("Mel band #")
    plt.xlabel("Frame #")
    plt.show()


def normalize_features(features):
	features = (features - np.mean(features))/np.std(features)
	return features


def write_to_csv(path):
	#loop through all instruments
	instrument_features = dict()
	nFiles = 0
	lpc_order = 15
	mel_order = 15

	for i in instruments:
		print('Getting features for ' + i)
		instrument_features[i] = []
		for r,d,f in os.walk(os.path.join(path,i)):
			for file in f:
				instrument_features[i].append(extract_features(os.path.join(r,file), mel_order, lpc_order))
				nFiles += 1
		

	print('...........................................')
	#number of rows - number of examples, number of columns = nfeatures + label
	#last column is label
	feature_array = np.zeros([nFiles,lpc_order+mel_order+1])
	n = 0

	for ins, features in instrument_features.iteritems():
		print('Writing features for ' + ins)
		for feature in features:
			feature_array[n,:-1] = feature
			feature_array[n,-1] = instruments.index(ins)
			n += 1

	saveTo = path + 'data_with_labels.csv'
	np.savetxt(saveTo, feature_array, delimiter=",") 




########################################################################################################################

def main():
	write_to_csv(trainPath)
	write_to_csv(testPath)









		




if __name__ == '__main__':
    main()