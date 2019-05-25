import numpy as np 
import os
from scipy.io.wavfile import read,write
import random
from shutil import copyfile


savePath = '../data/instruments'
trainPath = '../data/instruments/train'
testPath = '../data/instruments/test'
bachPath = "../Bach10_v1.1"
instruments = ['bassoon','violin','clarinet','saxphone']
nIns = 4


def break_into_parts(readpath,writepath,ins,id):
	(fs,audio) = read(readpath)
	t = np.arange(0,len(audio)-1)/fs
	winLen = 2 #length of each excerpt in seconds
	nFiles = np.floor(t[-1]/winLen)
	# print(nFiles)

	start = 0
	for n in range(int(nFiles)):
		excerpt = audio[start:start+(winLen*fs)]
		write(os.path.join(writepath,ins+str(id)+".wav"),fs,excerpt)
		start = start + winLen*fs 
		id += 1

	return id

def split_into_train_and_test(ins_path, ins, numFiles=161):
	#split 75:25
	nTrain = int(numFiles * 0.75)
	trp = os.path.join(trainPath,ins)
	tsp = os.path.join(testPath,ins)

	for r,d,files in os.walk(ins_path):
		random.shuffle(files)
		train = files[:nTrain]
		test = files[nTrain:]
		for ft in train:
			copyfile(os.path.join(ins_path,ft),os.path.join(trp,ft))
		for ft in test:
			copyfile(os.path.join(ins_path,ft),os.path.join(tsp,ft))


def main():
	songs = []
	if not os.path.exists(trainPath):
		os.mkdir(trainPath)
		os.mkdir(testPath)

	#get individual subdirectory for each song
	# r - root, d - directories, f - files
	for r,d,f in os.walk(bachPath):
		for song in d:
			if(song[0].isdigit()):
				songs.append(os.path.join(r,song))

	# print(songs)

	#go through all the files in each song directory
	#and get the wav files only
	sound_files = []
	for ins in instruments:
		id = 0
		for song in songs:
			# print(song)
			songname = song[song.rfind('/')+1:]
			# print(songname)
			for r,d,files in os.walk(song):
				for file in files:
					if(file.endswith(".wav") and ins in file):
						readPath = os.path.join(song,file)
						writePath = os.path.join(savePath,ins)
						if not os.path.exists(writePath):
							os.mkdir(writePath)
						id = break_into_parts(readPath,writePath,ins,id)
						print('Saved ' + file)

					
	#split excerpts into train and test files
	for ins in instruments:
		trpath = os.path.join(trainPath,ins)
		tspath = os.path.join(testPath,ins)
		if not os.path.exists(trpath):
			os.mkdir(trpath)
			os.mkdir(tspath)
		split_into_train_and_test(os.path.join(savePath,ins),ins)

if __name__ == '__main__':
    main()
