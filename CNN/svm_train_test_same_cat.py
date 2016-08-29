import numpy as np
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
import os, copy

"""
SVM trained and tested on the same category for each concept separately
"""

path_dat = "../../data"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = np.loadtxt(os.path.join(path_dat, "categories.txt"), type('str'))

lfc8 = np.load(os.path.join(path_dat, 'output_vgg19.npy')).reshape((len(prep), len(cats), 100, 1000))
labels = np.load(os.path.join(path_dat, 'labels_vgg19.npy')).reshape((len(prep), len(cats), 100))

nb_img = 100 

lran, lscore = [], []
for icon, concept in enumerate(prep):
    print(concept)
    for icat, category in enumerate(cats):
        print(category)
        X = lfc8[icon, icat]
        y = labels[icon, icat]
        # true scores
        skf = StratifiedKFold(y, 5)
        for train, test in skf:
             # train a svm on the output
            ytest = y[test]
            clf = svm.SVC()
            clf.fit(X[train], y[train])
            lscore.append(clf.score(X[test], y[test]))
            # permutation of the tests for the trained classifier 
            ytest = copy.deepcopy(y)[test] # because we shuffle afterwards
            for _ in range(X.shape[0]):
                np.random.shuffle(ytest)
                lran.append(clf.score(X[test], ytest))

lran = np.array(lran).reshape((len(prep), len(cats), -1, X.shape[0]))
lscore = np.array(lscore).reshape((len(prep), len(cats), -1))
ran = np.mean(lran, 2)
score = np.mean(lscore, 2)

np.save(os.path.join("../../data/res/", "train_test_same_cat_ran.npy"), ran)
np.save(os.path.join("../../data/res/", "train_test_same_cat_score.npy"), score)


