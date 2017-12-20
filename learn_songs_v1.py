from load_songs import *
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

#This python script will take pre-made python dictionaries with feature information about an artists songs, and do machine learning and data visualisation. When I load in songs and extract features in the other script, the data is stored in a nested dictinoary structure.

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
        feature_names=list(songfeat[0].keys()) #will be all our feature names
        features=[] #will be all our raw feature data
        for i in range(len(songfeat)):
            features.append(list(songfeat[i].values())) #take the songfeat dictionary and grab only the values (keys are just text labels for each feature)

        #create master lists of features and artists for the machine learning later
        all_features+=features
        all_artists+=artists
    return all_features, all_artists, feature_names




#Here we go, let's try some machine learning algorithms

if __name__ == '__main__':

    #load in all data saved from the feature extraction, *.pkl. Initiate figure and select colours
    path=sys.argv[1] #command line input is path to data
    all_data=glob.glob(path+'/*_data.pkl') #load in as many as you want

    colors = iter(cm.Set1(np.linspace(0, 1, len(all_data))))
    #colors = iter(cm.cubehelix(np.linspace(0, 1, len(all_data))))

    # load in artists with loads of songs - may or may not be splitting songs, try except:
    all_features, all_artists, feature_names = prepare_data(all_data) #feature names is same for all runs when unpacked (saves loading in a .pkl again)
    #Split our data into a training and testing data sets
    train_percent=float(sys.argv[2])
    # Test/train split as usual on artists with many songs
    features_train, features_test, artists_train, artists_test = train_test_split(all_features, all_artists, train_size=train_percent, random_state=0, stratify=all_artists)

    #now data is prepared for machine learning
    try:
        if len(artists_test)==len(features_test) and len(artists_train)==len(features_train):
            None
    except:
        print('artists and features are not same length: {0} != {1}',format(artists_test,features_test,artists_train,features_train))
        sys.exit()

    feature_names_flatten=np.array(feature_names).flatten()
    feature_names=np.transpose(feature_names)
    #print(np.transpose(feature_names))
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
    print(names)
    # Now try some classifiers!



    # Set up neutral network classifier

    from sklearn.preprocessing import StandardScaler
    nn_features_train=X_train
    nn_features_test=X_test
    scaler = StandardScaler()
    # normalise data and remove mean:
    scaler.fit(features_train)
    nn_features_train = scaler.transform(nn_features_train)
    nn_features_test = scaler.transform(nn_features_test)

    from sklearn.neural_network import MLPClassifier
    #not sure how many nodes to use? loop over some values until you're sure you've maxed the accuracy- 5000 is good for this:
    for i in range(5000,5001,1000):
        nn=MLPClassifier(hidden_layer_sizes=(i, ),solver='adam',max_iter=2000)
        nn.fit(nn_features_train, artists_train)
        nn_pred=nn.predict(nn_features_test)
        print('--'*30)
        print('MLP nn classifier with {0} hidden layers'.format(i))
        print(classification_report(artists_test, nn_pred,target_names=names))
        print('--'*30)

    '''
    #we could try SVC; it's quite poor compared to random forests.
    clf = svm.SVC(class_weight='balanced')
    clf.fit(X_train, Y_train_n)
    artists_SVM_pred=clf.predict(X_test)
    print('--'*30)
    print('SVM report:')
    print(classification_report(Y_test_n, artists_SVM_pred,target_names=names))
    print('--'*30)
    '''

    # Build a forest and compute the feature importances
    n_estimators=1000 #number of trees?
    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=2,class_weight='balanced')
    forest.fit(features_train, artists_train)
    artists_pred = forest.predict(features_test)
    accuracy_before=(accuracy_score(artists_test, artists_pred)) #we'll print this later as a comparison

    #Could check classification report here but we'll do this later after feature pruning
    #print('--'*30)
    #print('Random Forest report before feature pruning:')
    #print(classification_report(artists_test, forest.predict(features_test),target_names=names))
    #print('--'*30)

    #you could loop over trees to find out how many before you accuracy maxes output
    '''
    #how many trees are required? loop through values to find out
    scores=[]
    for val in range(1,100,10): #set accordingly
        clf=RandomForestClassifier(n_estimators=val,class_weight='balanced')
        validated = cross_validate(clf,X,Y,cv=5,scoring=['f1_weighted'])
        scores.append(validated)

    #make a nice plot:
    for i in range(0,len(scores)):
        print(scores[i]['test_f1_weighted'])
    y=[]
    x=[]
    e=[]
    for i in range(0,len(scores)):
        x.append(i)
        y.append(np.mean(scores[i]['test_f1_weighted']))
        e.append(np.mean(np.std(scores[i]['test_f1_weighted'])))
        print(np.mean(scores[i]['test_f1_weighted']), np.std(scores[i]['test_f1_weighted']))
    plt.errorbar(x,y,e)
    plt.show()
    '''


    ######################################################
    # Now lets repeat using a more streamlined pipeline to first remove unimportant features, then run a classifier on remaining ones.
    #we may want to try different classifiers and feature selection processes
    #an important note is that the pipeline automatically creates new feature data after removing pruned features.

    #first choose a model to prune features, then put it in pipeline - there are many we could try
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features_train, artists_train)
    rfc=RandomForestClassifier(n_estimators=n_estimators,random_state=2)
    modelselect='rfc' #set accordingly
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(rfc)),
        ('classification', RandomForestClassifier(n_estimators=n_estimators,random_state=2,class_weight='balanced'))
    ])
    #do the fit and feature selection
    pipeline.fit(features_train, artists_train)
    # check accuracy and other metrics:
    artists_important_pred = pipeline.predict(features_test)
    accuracy_after=(accuracy_score(artists_test, artists_important_pred))

    print('accuracy before pruning features: {0:.2f}'.format(accuracy_before))
    print('accuracy after pruning features: {0:.2f}'.format(accuracy_after))
    print('We should check other metrics for a full picture of this model:')
    print('--'*30)
    print('Random Forest report after feature pruning:')
    print(classification_report(artists_test, artists_important_pred,target_names=names))
    print('--'*30)




    #Now make plot of feature importances with standard deviations
    clf=pipeline.steps[1][1] #get classifier used
    importances = pipeline.steps[1][1].feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Now we've pruned bad features, create new feature_names_importanceorder_pruned array
    # Print the feature ranking if you want, but graph is nicer
    #print("Feature ranking:")
    feature_names_importanceorder_pruned=[]
    for f in range(len(indices)):
        #print("%d. feature %d (%f) {0}" % (f + 1, indices[f], importances[indices[f]]), feature_names[indices[f]])
        feature_names_importanceorder_pruned.append(str(feature_names[indices[f]]))
    # Plot the feature importances of the forest
    plt.figure()
    try:
        plt.title("\n".join(wrap("Feature importances pruned with {0}. n_est={1}. Trained on {2}% of data. Accuracy before={3:.3f}, accuracy after={4:.3f}".format(modelselect,n_estimators,train_percent*100,accuracy_before,accuracy_after,40))))
    except: #having issues with a fancy title?
        plt.title('After pruning features:')
    plt.bar(range(len(indices)), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.xticks(range(len(indices)), feature_names_importanceorder_pruned, rotation='vertical')
    plt.tight_layout()
    plt.show()

    # see which features were removed
    no_features=len(feature_names_importanceorder_pruned)
    print('Started with {0} features, now using {1}'.format(len(feature_names), no_features))
    print('features used were:')
    print( set(feature_names_flatten)-set(feature_names_importanceorder_pruned) )

    # plot confusion matrix - code adapted from sklearn manual page
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Showing normalized confusion matrix")
        else:
            print('Showing confusion matrix, without normalization')
        #print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    cnf_matrix = confusion_matrix(artists_test, artists_important_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=names,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=names, normalize=True,
    #                      title='Normalized confusion matrix')

    plt.show()
