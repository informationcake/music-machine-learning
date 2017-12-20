# music-machine-learning
[![DOI](https://zenodo.org/badge/110278467.svg)](https://zenodo.org/badge/latestdoi/110278467)

This website: https://informationcake.github.io/music-machine-learning/

This project is all about using python to extract features from audio waveforms, and then running machine learning algorithms to cluster and quantify music.

You will need to install the wonderful python library called Librosa, which deals with the handling of audio files. You'll also need the Python library called bokeh, used to create the interactive html plots. All other depenencies should be standard for regular python users.

load_songs.py loads in audio and performs feature extraction, saving the results to disk. The input is a single folder, usually named after the artist, containing only music files (mp3,wav,wma,mp4,etc...). e.g. python load_songs.py my_favourite_artist

plot_cluster_bokeh.py will create the interactive plot shown here using t-SNE or SVD, have a play!

t-SNE plots:
[All Artist Set 1](plots/plot_cluster_ManyArtists.md),
[All Artist Set 2](plots/plot_cluster_ManyArtists2.md),
[The Flashbulb 1](plots/TheFlashbulb_TSNE.md),
[The Flashbulb 2](plots/TheFlashbulb_TSNE2.md),
[Avril Lavigne 1](plots/AvrilLavigne_TSNE_run1.md),
[Avril Lavigne 2](plots/AvrilLavigne_TSNE_run2.md)

SVD plots:
[All Artists](plots/SVD_artists_plot.md)

plot_similarity.py will create a plot of the similarity matrix, averaging over all an artists songs.

learn_songs_v0.py will take the _data.pkl files output from load_songs.py, and perform some machine learning and data visualisation techniques. v0 is a blank version you can start from scratch yourself (if you know how to implement machine learning). 

learn_songs_v1.py is a version which has some machine learning code added in already. You can run it and see what happens, tweak it, exploring parts I've commented out. Hopefully it will be useful for anyone wanting to explore how to understand implementing machine learning.

I will add more info as I develop this. I'm quite a bit further ahead in this project than this github repo suggests, as I'm only uploading code once I'm sure it will be useful for others. 

You can read in a bit more depth about what is happening on my Google site [informationcake.com](https://sites.google.com/view/informationcake/music/machine-learning) where I show some results and plots.

Thank you for your interest, and if you have ideas, do let me know!

Cheers,
Alex
