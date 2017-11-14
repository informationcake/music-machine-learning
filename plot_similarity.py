from load_songs import *
import sys
import glob
import matplotlib.cm as cm
import scipy.stats as stats
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

#This script will create the similarity matrix, average over an artists songs. I just adapted it from my other scripts so some parts of it are redundant.

#load in all data saved from the feature extraction, *.pkl. Initiate figure and select colours
path=sys.argv[1]
all_data=glob.glob(path+'/*.pkl')

#As i did feature extraction on each artist seperately, loop through them. Create lists of song names and features
all_features_avg=[]
all_artists=[]
for artist in all_data:
    data=load_obj(artist.replace('.pkl',''))
    print('loading {0}'.format(artist))
    songname=[] #will be a list of song names
    songfeat=[] #will be a list of dictionaries containing the song feature data
    artists=[] #will be a list of artists, as I load them artist at a time should be straightforward.
    for song in data: #data is a dictionary, keys are song names
        songfeat.append(data[song]) #data corresponding to each dictionary key is another dictionary with 25 features and labels
        songname.append(song)
    artists.append(artist.replace('_data.pkl','').replace('all_','').replace(path,''))

    features=[] #will be all our raw feature data
    feature_names=[] #will be all our feature names
    feature_names.append(list(songfeat[0].keys()))
    for i in range(len(songfeat)):
        features.append(list(songfeat[i].values())) #take the songfeat dictionary and grab only the values (keys are just text labels for each feature)

    #create master lists of artists
    all_artists+=artists
    # Now we compute artists simliarity based on the distance between their songs on average in the 29 dimensional feature space
    features_avg=np.mean(features, axis=0)
    all_features_avg.append(features_avg)


#calculate distance between artists in 29 dimensional features space
print(np.shape(all_features_avg))
dist=pdist(all_features_avg)
dist_matrix=squareform(dist)

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
x=range(0,len(all_artists))
i=ax.imshow(dist_matrix, cmap='hot', interpolation='nearest')
plt.xticks(x,all_artists,rotation='vertical')
plt.yticks(x,all_artists)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
fig.colorbar(i, orientation='vertical')
plt.tight_layout()
plt.savefig('Artist Similarity Matrix')
plt.show()
