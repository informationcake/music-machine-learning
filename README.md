# music-machine-learning
Using python to extract features from audio waveforms, and then running machine learning algorithms.

You will need to install the wonderful python library called Librosa, which deals with the handling of audio files. You'll also need the Python library called bokeh, used to create the interactive html plots. All other depenencies should be standard for regular python users.

load_songs.py loads in audio and performs feature extraction, saving the results to disk. The input is a single folder, usually named after the artist, containing only music files (mp3,wav,wma,mp4,etc...). e.g. python load_songs.py my_favourite_artist

plot_cluster_bokeh.py will create the interactive plot shown here, have a play!

https://informationcake.github.io/music-machine-learning/

learn_songs.py will take the _data.pkl files output from load_songs.py, and perform some machine learning and data visualisation techniques. I have yet to upload this part to this repo, watch this space.

I will add more info as I develop this. I'm quite a bit further ahead in this project than this github repo suggests, as I'm only uploading code once I'm sure it will be useful for others. You can get a better feel for this project on my webpage where I show some results and what will be coming on this repo.

You can read in a bit more depth about what is happening here:

https://sites.google.com/view/informationcake/music/machine-learning

Thank you for your interest, and if you have ideas, do let me know!

Cheers,
Alex
