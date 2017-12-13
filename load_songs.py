# -*- coding: utf-8 -*-
import librosa.display
import librosa as lb
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import multiprocessing
import itertools
import sys
from collections import OrderedDict
from  more_itertools import unique_everseen
from scipy.stats import skew
from scipy.stats import kurtosis

#This python script will load in songs and extract features from the waveform. It will then create a dictionary of all the results, ready for plotting in another script.
#At the top we have a load of functions pre-defined, skip down to __main__ to see the steps we run

#a function to split up a song into TIME chunks
def splitT(mint,maxt,songdat):
    splittime=[]
    for i in range(mint,maxt):
        splittime.append(songdat[:,i]) # first axis is freq, second axis is time. Return all freq for specific time range.
    return (np.array(splittime))

#a function to split up a song into FREQ chunks
def splitF(minv, maxv, songdat):
    splitfreq = []
    for i in range(minv,maxv):
        splitfreq.append(songdat[i,:]) # first axis is freq, second axis is time. Return all time for specific freq range.
    return (np.array(splitfreq))

#This is the main function which gets features from the songs. Most values returned are the mean of the whole time series, hence '_a'.
def get_features_mean(song,sr,hop_length,n_fft):
    try:
        print('extracting features...')
        y_harmonic, y_percussive = lb.effects.hpss(song) #split song into harmonic and percussive parts
        stft_harmonic=lb.core.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length)	#Compute power spectrogram.
        stft_percussive=lb.core.stft(y_percussive, n_fft=n_fft, hop_length=hop_length)	#Compute power spectrogram.
        #stft_all=lb.core.stft(song, n_fft=n_fft, hop_length=hop_length)	#Compute power spectrogram.
        band_resolution=[5] #[5,25] Choose number of bands, do low and high resolution?
        bands_dict=OrderedDict()
        for no_bands in band_resolution:
            bands=np.logspace(1.3,4,no_bands)/10 #note that as n_fft is 2050 (I've decided this is sensible resolution), bands/10=freq
            bands_int=bands.astype(int)
            bands_int_unique=list(unique_everseen(bands_int)) #removing double entries less than 100Hz, because logspace bunches up down there and we don't need doubles when rounding to the nearest 10 Hz.
            for i in range(0,len(bands_int_unique)-1):
                _h=lb.feature.rmse(y=(splitF(bands_int_unique[i],bands_int_unique[i+1],stft_harmonic)))
                _p=lb.feature.rmse(y=(splitF(bands_int_unique[i],bands_int_unique[i+1],stft_percussive)))
                #Calculate statistics for harmoinc and percussive over the time series.
                rms_h=np.mean(np.abs(_h))
                std_h=np.std(np.abs(_h))
                skew_h=skew(np.mean(np.abs(_h), axis=0))  #skew of the time series (avg along freq axis, axis=0)
                kurtosis_h=kurtosis(np.mean(np.abs(_h), axis=0), fisher=True, bias=True) #kurtosis of time series (avg along freq axis=0)
                rms_p=np.mean(np.abs(_p))
                std_p=np.std(np.abs(_p))
                skew_p=skew(np.mean(np.abs(_p), axis=0))  #skew of the time series (avg along freq axis, axis=0)
                kurtosis_p=kurtosis(np.mean(np.abs(_p), axis=0), fisher=True, bias=True) #kurtosis of time series (avg along freq axis=0)

                #Append results to dict, with numbers as band labels
                bands_dict.update({'{0}band_rms_h{1}'.format(no_bands,i):rms_h,'{0}band_rms_p{1}'.format(no_bands,i):rms_p})
                bands_dict.update({'{0}band_std_h{1}'.format(no_bands,i):std_h,'{0}band_std_p{1}'.format(no_bands,i):std_p})
                bands_dict.update({'{0}band_skew_h{1}'.format(no_bands,i):skew_h,'{0}band_skew_p{1}'.format(no_bands,i):skew_p})
                bands_dict.update({'{0}band_kurtosis_h{1}'.format(no_bands,i):kurtosis_h,'{0}band_kurtosis_p{1}'.format(no_bands,i):kurtosis_p})

        #stft=lb.feature.chroma_stft(song, sr, n_fft=n_fft, hop_length=hop_length)	#Compute a chromagram from a waveform or power spectrogram.
        #stft_a=np.mean(stft[0])
        #stft_std=np.std(stft[0])
        #rmse=lb.feature.rmse(y=song)	#Compute root-mean-square (RMS) energy for each frame, either from the audio samples y or from a spectrogram S.
        #rmse_a=np.mean(rmse)
        #rmse_std=np.std(rmse)
        rmseH=np.abs(lb.feature.rmse(y=stft_harmonic))	#Compute root-mean-square (RMS) energy for harmonic
        rmseH_a=np.mean(rmseH)
        rmseH_std=np.std(rmseH)
        rmseH_skew=skew(np.mean(rmseH, axis=0))
        rmseH_kurtosis=kurtosis(np.mean(rmseH, axis=0), fisher=True, bias=True)

        rmseP=np.abs(lb.feature.rmse(y=stft_percussive))	#Compute root-mean-square (RMS) energy for percussive
        rmseP_a=np.mean(rmseP)
        rmseP_std=np.std(rmseP)
        rmseP_skew=skew(np.mean(rmseP, axis=0))
        rmseP_kurtosis=kurtosis(np.mean(rmseP, axis=0), fisher=True, bias=True)

        centroid=lb.feature.spectral_centroid(song, sr, n_fft=n_fft, hop_length=hop_length)	#Compute the spectral centroid.
        centroid_a=np.mean(centroid)
        centroid_std=np.std(centroid)
        bw=lb.feature.spectral_bandwidth(song, sr, n_fft=n_fft, hop_length=hop_length)	#Compute p’th-order spectral bandwidth:
        bw_a=np.mean(bw)
        bw_std=np.std(bw)
        contrast=lb.feature.spectral_contrast(song, sr, n_fft=n_fft, hop_length=hop_length)	#Compute spectral contrast [R16]
        contrast_a=np.mean(contrast)
        contrast_std=np.std(contrast)
        polyfeat=lb.feature.poly_features(y_harmonic, sr, n_fft=n_fft, hop_length=hop_length)	#Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
        polyfeat_a=np.mean(polyfeat[0])
        polyfeat_std=np.std(polyfeat[0])
        tonnetz=lb.feature.tonnetz(librosa.effects.harmonic(y_harmonic), sr)	#Computes the tonal centroid features (tonnetz), following the method of [R17].
        tonnetz_a=np.mean(tonnetz)
        tonnetz_std=np.std(tonnetz)
        zcr=lb.feature.zero_crossing_rate(song, sr, hop_length=hop_length)  #zero crossing rate
        zcr_a=np.mean(zcr)
        zcr_std=np.std(zcr)
        onset_env=lb.onset.onset_strength(y_percussive, sr=sr)
        onset_a=np.mean(onset_env)
        onset_std=np.std(onset_env)
        D = librosa.stft(song)
        times = librosa.frames_to_time(np.arange(D.shape[1])) #not returned, but could be if you want to plot things as a time series
        bpm,beats=lb.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env, units='time')
        beats_a=np.mean(beats)
        beats_std=np.std(beats)

        features_dict=OrderedDict({'rmseP_a':rmseP_a,'rmseP_std':rmseP_std,'rmseH_a':rmseH_a,'rmseH_std':rmseH_std,'centroid_a':centroid_a,'centroid_std':centroid_std,'bw_a':bw_a,'bw_std':bw_std,'contrast_a':contrast_a,'contrast_std':contrast_std,'polyfeat_a':polyfeat_a,'polyfeat_std':polyfeat_std,'tonnetz_a':tonnetz_a,'tonnetz_std':tonnetz_std,'zcr_a':zcr_a,'zcr_std':zcr_std,'onset_a':onset_a,'onset_std':onset_std,'bpm':bpm, 'rmseP_skew':rmseP_skew, 'rmseP_kurtosis':rmseP_kurtosis, 'rmseH_skew':rmseH_skew, 'rmseH_kurtosis':rmseH_kurtosis})

        combine_features={**features_dict,**bands_dict}
        print('features extracted successfully')
        return combine_features

    except:
        print('.'*20+'FAILED'+'.'*20)
        print('.'*40)

#a function to look at beat tracking... not used in machine learning yet, just random investigations.
def beattrack(song,sr,hop_length,n_fft):
    y_harmonic, y_percussive = lb.effects.hpss(song)
    beattrack=lb.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=None, hop_length=hop_length, start_bpm=120.0, tightness=100, trim=True, bpm=None, units='frames')

#load music function, accepts any format i've encountered: mp3,wav,wma bla bla
def load_music(songname1,songpath1):
    try:
        print('loading the song: {0} ......... located here: {1} '.format(songname1, songpath1))
        songdata1, sr1 = lb.load(songpath1) #librosa library used to grab songdata and sample rate
        print ('done........ '+songname1)
        return [songname1,songdata1,sr1]
    except: #the song could be corrupt? you could be trying to load something which isn't a song?
        print('..............................FAILED...............................')
        print(songpath1)
        print('...................................................................')

#functions for saving/loading the python dictionaries to disk
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#If you want a grid-plot to test anything out, this will help. Although I've made sure get_features returns only averaged values, not time-series data, so meh.
def gridplot(data_dict,feature,size,N,ind):
    f, axarr = plt.subplots(size, size, sharey=True)
    i=0
    j=0
    for key in data_dict:
        #print (i,j)
        axarr[i,j].plot(np.convolve(data_dict[key][feature][ind],np.ones((N,))/N, mode='valid'))
        axarr[i, j].set_title(key[:3])
        if j==size-1: i+=1
        j=0 if j==size-1 else j+1
    for i in range(1,size,1):
        plt.setp([a.get_yticklabels() for a in axarr[:, i]], visible=False)
    plt.savefig('test.png')







#OK so here we go...

if __name__ == "__main__":
        start_load=time.time() #we're going to want know how long this takes...
        num_workers = multiprocessing.cpu_count() #if we don't use multiple cores we may as well give up now. This is how many your computer has.
        print('you have {0} cores available to do your bidding...'.format(num_workers))
        num_workers=32 #I was playing around with changing this
        #multi=int(sys.argv[1]) #at one point I was testing this as a command line input
        n_fft1=2050 #important parameter here; this is the size of the fft window. these are sensible values
        hop_length1=441 #n_fft/5 is a sensisble value. too large and you don't sample properly.

        #create song database, songdb:
        songname_tmp=[]
        songpath_tmp=[]
        #path='./audio_alex/'
        path=sys.argv[1] #the only command line input is the path to the folder of music
        print(path)
        savefile=str(path)+'_data' #it's saved with the same folder name but with _data.pkl on the end.
        #now load song data in
        for song in os.listdir(path):
            #print (song)
            songname_tmp.append(song)
            songpath_tmp.append(path+'/'+song)

        #print(songname)
        songname=songname_tmp #i'm just reassigning the name incase of tests with commented out lines...
        songpath=songpath_tmp
        #if you want to test this on a small number of songs first (e.g. 32), replace previous two lines with the following:
        #songname=songname_tmp[:31] #remember indices starts at zero.
        #songname=songname_tmp[:31]

        print('loading songs...')
        #Here we go with multi-processing, loading all our song data in
        with multiprocessing.Pool(processes=num_workers) as pool:
            songdb=pool.starmap(load_music,zip(songname,songpath)) #btw a starmap is a way to pass multiple arguments to a function using multi-process
            pool.close()
            pool.join()
        print('finished loading songs into songdb')
        #print (songdb)
        print ('loaded {0} songs into memory'.format(len(songdb)))
        songdb=[x for x in songdb if x is not None] #remove any entries where loading may have failed for any reason (rare cases)
        #parse song data to individual lists ready for feature extraction function (because we can't slice nested lists)
        song_name=[] #text
        song_data=[] #list of numbers
        song_sr=[] #sample rate
        for song1 in songdb: song_name.append(song1[0])
        for song1 in songdb: song_data.append(song1[1])
        for song1 in songdb: song_sr.append(song1[2])

        start_feat = time.time() #note the time
        print("Data is all ready, now extracting features from the songs...")
        #extract features from songs with multiprocesssing
        with multiprocessing.Pool(processes=num_workers,maxtasksperchild=1) as pool:
            res=pool.starmap(get_features_mean,zip(song_data,song_sr,itertools.repeat(hop_length1),itertools.repeat(n_fft1)))
            pool.close()
            pool.join()

        #concatenate each songs features (res) into dictionary
        print('concatenating results into a massive dictionary...')
        data_dict_mean={}
        for i in range(0,len(songdb)):
            data_dict_mean.update({song_name[i]:res[i]})

        #print features to screen to check
        print('The features extracted from the songs are: ')
        print(res[0].keys())
        print('saving dictionary to disk...')
        save_obj(data_dict_mean,savefile)
        end_feat=time.time() #note finish time
        print("loading time: {0} seconds".format(start_feat-start_load))
        print("feature extraction time: {0} seconds".format(end_feat-start_feat))
        print("total time: {0} seconds".format(end_feat-start_load))
        print('finished')
