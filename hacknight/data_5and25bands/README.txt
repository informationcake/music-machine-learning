Here are the .pkl files containing the dictinoaries created by load_songs.py

I used no_bands=[5,25], so the bands are on a log scale: bands=np.logspace(1.3,4,no_bands)/10

Hence we have both a course (5 band) and fine (25 band) resolution in frequency.

learn_songs_v0.py takes these .pkl files as input
