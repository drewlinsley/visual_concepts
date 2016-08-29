import numpy as np
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
import os, copy

"""
For each category, training on one concept and testing on another one
"""

path_dat = "../../data"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = np.loadtxt(os.path.join(path_dat, "categories.txt"), type('str'))
lfc8 = np.load(os.path.join(path_dat, 'output_vgg19.npy')).reshape((len(prep), len(cats), 100, 1000))
labels = np.load(os.path.join(path_dat, 'labels_vgg19.npy')).reshape((len(prep), len(cats), 100))
nb_img = 50 

for icat, cat in enumerate(cats):
    category = cats[icat]
    print(category)
    lran, lscore = [], [] 
    for icon, concepti in enumerate(prep):
        for jcon, conceptj in enumerate(prep):
            print(concepti)
            print(conceptj)
            Xi = lfc8[icon, icat, :nb_img]
            Xj = lfc8[jcon, icat, :nb_img]
            X = np.vstack([Xi, Xj])
            y = np.hstack((np.ones((nb_img, )), np.zeros((nb_img,)))) 
            # true scores
            skf = StratifiedKFold(y, 5)
            true_score = []
            for train, test in skf:
                 # train a svm on the output
                clf = svm.SVC()
                clf.fit(X[train], y[train])
                lscore.append(clf.score(X[test], y[test]))
                ytest = copy.deepcopy(y)[test] # because we shuffle afterwards
                for i in range(X.shape[0]):
                    np.random.shuffle(ytest)
                    lran.append(clf.score(X[test], ytest))

    lran = np.array(lran).reshape((len(prep), len(prep), -1, X.shape[0]))
    lscore = np.array(lscore).reshape((len(prep), len(prep), -1))
    ran = np.mean(lran, 2)
    score = np.mean(lscore, 2)

    np.save(os.path.join("../../data/res/", cat + "_across_concepts_ran.npy"), ran)
    np.save(os.path.join("../../data/res/", cat + "_across_concepts_score.npy"), score)


