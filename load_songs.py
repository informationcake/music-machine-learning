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

#This python script will load in songs and extract features from the waveform. It will then create a dictionary of all the results, ready for plotting in another script.
#At the top we have a load of functions pre-defined, skip down to __main__ to see the steps we run

#a function to split up a song into freq chunks
def splitF(minv, maxv, songdat):
    splitfreq = []
    for i in range(minv,maxv):
        splitfreq.append(songdat[i])
    return (np.array(splitfreq))

#This is the main function which gets features from the songs. Most values returned are the mean of the whole time series, hence '_a'.
def get_features_mean(song,sr,hop_length,n_fft):
    try:
        print('extracting features on...')
        y_harmonic, y_percussive = lb.effects.hpss(song)
        stft_core=lb.core.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length)	#Compute power spectrogram.
        rms_b1=np.mean(lb.feature.rmse(y=(splitF(0,10,stft_core)))) #RMS and STD of energy in different spectral bands
        std_b1=np.std(lb.feature.rmse(y=(splitF(0,10,stft_core))))
        rms_b2=np.mean(lb.feature.rmse(y=(splitF(10,50,stft_core))))
        std_b2=np.std(lb.feature.rmse(y=(splitF(10,50,stft_core))))
        rms_b3=np.mean(lb.feature.rmse(y=(splitF(50,100,stft_core))))
        std_b3=np.std(lb.feature.rmse(y=(splitF(50,100,stft_core))))
        rms_b4=np.mean(lb.feature.rmse(y=(splitF(100,400,stft_core))))
        std_b4=np.std(lb.feature.rmse(y=(splitF(100,400,stft_core))))
        rms_b5=np.mean(lb.feature.rmse(y=(splitF(400,1026,stft_core))))
        std_b5=np.std(lb.feature.rmse(y=(splitF(400,1026,stft_core))))
        rmseH=lb.feature.rmse(y=y_harmonic)	#Compute root-mean-square (RMS) energy for each frame, either from the audio samples y or from a spectrogram S.
        rmseH_a=np.mean(rmseH)
        rmseH_std=np.std(rmseH)
        rmseP=lb.feature.rmse(y=y_percussive)	#Compute root-mean-square (RMS) energy for each frame, either from the audio samples y or from a spectrogram S.
        rmseP_a=np.mean(rmseP)
        rmseP_std=np.std(rmseP)
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
        #D = librosa.stft(song)
        #times = librosa.frames_to_time(np.arange(D.shape[1])) #not returned, but could be if you want to plot things as a time series
        bpm,beats=lb.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env, units='time')
        beats_a=np.mean(beats)
        beats_std=np.std(beats)
        print('features extracted successfully')
        return {'rms_b1':rms_b1,'std_b1':std_b1,'rms_b2':rms_b2,'std_b2':std_b2,'rms_b3':rms_b3,'std_b3':std_b3,'rms_b4':rms_b4,'std_b4':std_b4,'rms_b5':rms_b5,'std_b5':std_b5,'rmseP_a':rmseP_a,'rmseP_std':rmseP_std,'rmseH_a':rmseH_a,'rmseH_std':rmseH_std,'centroid_a':centroid_a,'centroid_std':centroid_std,'bw_a':bw_a,'bw_std':bw_std,'contrast_a':contrast_a,'contrast_std':contrast_std,'polyfeat_a':polyfeat_a,'polyfeat_std':polyfeat_std,'tonnetz_a':tonnetz_a,'tonnetz_std':tonnetz_std,'zcr_a':zcr_a,'zcr_std':zcr_std,'onset_a':onset_a,'onset_std':onset_std,'bpm':bpm}
    except:
        print('.'*20+song+' FAILED'+'.'*20)
        print('.'*40)


#a function to look at beat tracking... not used in machine learning yet, just random investigations.
def beattrack(song,sr,hop_length,n_fft):
    y_harmonic, y_percussive = lb.effects.hpss(song)
    beattrack=lb.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=None, hop_length=hop_length, start_bpm=120.0, tightness=100, trim=True, bpm=None, units='frames')


#load music function, accepts any format i've encountered: mp3,mp4,wav,wma bla bla
def load_music(songname1,songpath1):
    try:
        print('loading the song: {0} ......... located: {1} '.format(songname1, songpath1))
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




#Here we go...

if __name__ == "__main__":
        start_load=time.time() #we're going to want know how long this takes...
        num_workers = multiprocessing.cpu_count() #if we don't use multiple cores we may as well give up now. This is how many your computer has.
        print('you have {0} cores available to do your bidding...'.format(num_workers))
        #multi=int(sys.argv[1]) #at one point I was testing this as a command line input
        n_fft1=2050 #important parameter here; this is the size of the fft window. these are sensible values, if anything make them larger?
        hop_length1=441 #n_fft/5 is a sensisble value. too large and you don't sample properly.
        #songdb=load_obj('songdb')

        #create song database, songdb:
        songname_tmp=[]
        songpath_tmp=[]
        #path='./audio_alex/'
        path=sys.argv[1] #the only command line input is the path to the folder of music
        savefile=str(path)+'_data' #it's saved with the same folder name but with _data.pkl on the end.
        #now load song data in
        for song in os.listdir(path):
            #print (song)
            songname_tmp.append(song)
            songpath_tmp.append(path+'/'+song)

        songname=songname_tmp #i'm just reassigning the name incase of tests with commented out lines...
        songpath=songpath_tmp
        #if you want to test this on a small number of songs first (e.g. 32, HIGHLY RECOMMENDED), replace previous two lines with the following:
        #songname=songname_tmp[:31] #remember indices starts at zero.
        #songname=songname_tmp[:31]

        #Here i was testing it on one song, before I was sure that multi-processing was working
        #print('testing on one song......')
        #dummy=load_music_mt(songname[0],songpath[0]) #first entry in songname and songpath lists
        #print(dummy)
        #pool=multiprocessing.Pool(processes=8)
        #songdb=pool.starmap(load_music,zip(songname,songpath))
        #print('test done')

        #Here we go with multi-processing, loading all our song data in
        with multiprocessing.Pool(processes=num_workers) as pool:
            songdb=pool.starmap(load_music,zip(songname,songpath)) #btw a starmap is a way to pass multiple arguments to a function using multi-process
            pool.close()
            pool.join()

        print('finished loading songs into songdb')
        #print (songdb)
        print ('loaded {0} songs into memory'.format(len(songdb)))
        songdb=[x for x in songdb if x is not None] #remove any entries where features may have failed for any reason (rare cases)
        #parse song data to individual lists ready for feature extraction function (because nested lists make things awkward)
        song_name=[] #text
        song_data=[] #list of numbers
        song_sr=[] #sample rate
        for song1 in songdb: song_name.append(song1[0])
        for song1 in songdb: song_data.append(song1[1])
        for song1 in songdb: song_sr.append(song1[2])

        #test feature extraction on one song before multi process
        #print('testing......')
        #dummy=get_features_mean(song_data[0],song_sr[0],hop_length1,n_fft1)
        #print('test done')
        #print('.'*40)

        start_feat = time.time() #note the time
        print("Data is all ready, now extracting features from the songs...")
        #extract features from songs with multiprocesssing
        with multiprocessing.Pool(processes=num_workers,maxtasksperchild=1) as pool:
            res=pool.starmap(get_features_mean,zip(song_data,song_sr,itertools.repeat(hop_length1),itertools.repeat(n_fft1)))
            pool.close()
            pool.join()

        #concatenate each result into dictionary
        print('concatenating results into a massive dictionary...')
        data_dict_mean={}
        for i in range(0,len(songdb)):
            data_dict_mean.update({song_name[i]:res[i]})

        print('saving dictionary to disk...')
        save_obj(data_dict_mean,savefile)
        end_feat=time.time() #note finish time
        print("loading time: {0} seconds".format(start_feat-start_load))
        print("feature extraction time: {0} seconds".format(end_feat-start_feat))
        print("total time: {0} seconds".format(end_feat-start_load))
        print('finished')

#the _data.pkl saved to disk is now ready to pass to any other code. I will soon add my machine learning code which uses this data.
