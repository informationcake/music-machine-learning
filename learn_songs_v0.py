from load_songs import * #incase you want to use the other functions.
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import sklearn
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import sys
import glob
import matplotlib.cm as cm
from textwrap import wrap
import scipy.stats as stats

#This python script will take pre-made python dictionaries with feature information about an artists songs, and do machine learning and data visualisation. load_songs.py stored the data in a nested dictinoary structure. Keys are song names, values are dictionaries where keys are feature names and values are feature values. i.e. two dictionary levels.

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#function to load in data from load_songs script.
def prepare_data(all_data_in):
    all_features=[]
    all_artists=[]
    for artist in all_data_in: #As i did feature extraction on each artist seperately, loop through them. Create lists of song names and features
        data=load_obj(artist.replace('.pkl',''))
        print('loading {0}'.format(artist))
        songname=[] #will be a list of song names
        songfeat=[] #will be a list of dictionaries containing the song feature data
        artists=[] #will be a list of artists, as I load them artist at a time should be straightforward.
        for song in data: #data is a dictionary, keys are song names
            songfeat.append(data[song]) #data corresponding to each dictionary key is another dict with features and labels
            songname.append(song)
            artists.append(artist.replace('_data.pkl','').replace('all_','').replace(path,'').replace('_data_testsplit.pkl','').replace('_data_trainsplit.pkl',''))

        #if we want to modify the features, could do it here
        #e.g. removing some features
        '''
        print (len(songfeat[0]))
        for i in range(len(songfeat)):
            for k in ['onset_a','onset_std','bpm','centroid_a','centroid_std','polyfeat_a','polyfeat_std','zcr_a','zcr_std']:
                songfeat[i].pop(k,None)

        print (len(songfeat[0]))
        '''
        feature_names=[] #will be all our feature names
        feature_names.append(list(songfeat[0].keys()))
        features=[] #will be all our raw feature data
        for i in range(len(songfeat)):
            features.append(list(songfeat[i].values())) #take the songfeat dictionary and grab only the values (keys are just text labels for each feature)

        #create master lists of features and artists for the machine learning later
        all_features+=features
        all_artists+=artists
    return all_features, all_artists, feature_names



colors = iter(cm.Set1(np.linspace(0, 1, 10)))
#colors = iter(cm.cubehelix(np.linspace(0, 1, len(all_data))))

#load in all data saved from the feature extraction, *.pkl.
path=sys.argv[1] #command line input is path to data
all_data=glob.glob(path+'/*_data.pkl') #load in as many as you want

# load in artists with loads of songs:
all_features, all_artists, feature_names = prepare_data(all_data) #feature names is same for all runs when unpacked (saves loading in a .pkl again)
test_percent=float(sys.argv[2]) #second commandline input is test %
# Test/train split. Stratify to correct for inbalanced classes.
features_train, features_test, artists_train, artists_test = train_test_split(all_features, all_artists, test_size=test_percent, random_state=0, stratify=all_artists)

feature_names=np.transpose(feature_names)
print('Feature names are: ')
print(np.transpose(feature_names))

# set up data as numerical classes as well as labeled string classes - some classifiers require numbered labels, not artists as strings
X_test = np.array(features_test)
Y_test = np.array(artists_test)
le=preprocessing.LabelEncoder()
le.fit(Y_test)
Y_test_n=le.transform(Y_test)

X_train = np.array(features_train)
Y_train = np.array(artists_train)
le=preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train_n=le.transform(Y_train)
names=np.unique(Y_test)

# Now try some classifiers! Some reminders:
# Inputs to train classifiers are: features_train, artists_train
# Inputs to test classifiers are: features_test, artists_test
# Some classifiers will require classes labeled as numbers; if so, use Y_train_n, Y_test_n, X_test, X_train

#CODE BELOW IS JUST AN EXAMPLE, YOU WILL NEED TO EDIT IT A LOT DEPENDING ON WHAT YOU WANT TO DO




#investigate log-loss? Compare to dummy_classifier? gradient boosted forest? hyper-parms in your fav classifier? Confusion matrix for multi-class?


#e.g. lets do a simple random forest
forest=RandomForestClassifier(n_estimators=25, random_state=1, class_weight='balanced')
forest.fit(feature_train, artists_train)
predicted_artists=forest.predict(features_test)
print(Classification_report(artists_test,predicted_artists))

#Support Vector MAchines (SVM) need numerical classes:
svm=svm.SVC(class_weight='balanced')
svm.fit(X_train, Y_train_n)

#MPL? basic neural network. NEEDS normalised data. 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(features_train)
nn=MPLClassifier(hidden_layer_sizes=(i,), solver='adam', max_iter=2000)
#nn.fit()
#etc etc



#how many trees are needed? check out cross_validating stuff:
scores=[]
for trees in range(1,10):
    forest=RandomForestClassifier(n_estimators=25, random_state=1, class_weight='balanced')
    validated=cross_validate(forest, X, Y, cv=5, scoring=['f1_weighted'])
    score.append(validated)


    
    
#Rank features and remove unimportant ones?
genres_pred=forest.predict(features_test)
accuracy_before=accuracy_score(artists_test, artists_pred)
print('accuracy with all features included: {0}'.format(accuracy_before))
# get importances and standard deviations, plot them in order of importance
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
feature_names_importanceorder=[]
for f in range(len(indices)):
    #print("%d. feature %d (%f) {0}" % (f + 1, indices[f], importances[indices[f]]), feature_names[indices[f]])
    feature_names_importanceorder.append(str(feature_names[indices[f]]).rstrip("'][").strip("[]'"))

# plot the feature importances of the forest
plt.figure()
plt.title("Feature importances before pruning")
plt.bar(range(len(indices)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), indices)
plt.xlim([-1, len(indices)])
plt.xticks(range(len(indices)), feature_names_importanceorder, rotation='vertical')
plt.tight_layout()
plt.show()





