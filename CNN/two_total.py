import numpy as np
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
import os, copy

path_dat = "../../data"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = np.loadtxt(os.path.join(path_dat, "categories.txt"), type('str'))
lfc8 = np.load(os.path.join(path_dat, 'output_vgg19.npy')).reshape(len(prep), len(cats), 100, 1000)
labels = np.load(os.path.join(path_dat, 'labels_vgg19.npy')).reshape(len(prep), len(cats), 100)
nb_img = 100 

# fewer categories 
lran = np.zeros((len(prep), len(cats), len(cats), 100)) 
lscore = np.zeros((len(prep), len(cats), len(cats))) 
for icon, concepti in enumerate(prep):
    print(concepti)
    for icat, categoryi in enumerate(cats):
        for jcat, categoryj in enumerate(cats):
            print(categoryi)
            print(categoryj)
            Xtrain = lfc8[icon, icat, 0:nb_img]
            Xtest = lfc8[icon, jcat, 0:nb_img]
            ytrain = np.hstack((np.ones((nb_img/2, )), np.zeros((nb_img/2,)))) 
            ytest = np.hstack((np.ones((nb_img/2, )), np.zeros((nb_img/2,)))) 
            # true scores
            skf = StratifiedKFold(ytrain, 5)
            true_score = []
            for train, test in skf:
                 # train a svm on the output
                clf = svm.SVC()
                clf.fit(Xtrain[train], ytrain[train])
                true_score.append(clf.score(Xtest[test], ytest[test]))
            print(np.mean(true_score))
            lscore[icon, icat, jcat] = np.mean(true_score)
            for i in range(100):
                np.random.shuffle(ytrain)
                skf = StratifiedKFold(ytrain, 5)
                ran_score = []
                for train, test in skf:
                    # train a svm on the output
                    clf = svm.SVC()
                    clf.fit(Xtrain[train], ytrain[train])
                    ran_score.append(clf.score(Xtest[test], ytest[test]))
                lran[icon, icat, jcat, i] = np.mean(ran_score)
np.save(os.path.join("../../data/res/", "all_across_lran.npy"), lran)
np.save(os.path.join("../../data/res/", "all_across_score.npy"), lscore)


