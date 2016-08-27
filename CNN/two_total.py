import numpy as np
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
import os, copy

path_dat = "../../data"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = os.listdir(os.path.join(path_dat, "imgs"))
lfc8 = np.load(os.path.join(path_dat, 'output_vgg19.npy')).reshape(len(prep), len(cats), 100, 1000)
labels = np.load(os.path.join(path_dat, 'labels_vgg19.npy').reshape(len(prep), len(cats), 100))
nb_img = 50 

lran = np.zeros((len(prep), len(cats), len(cats), 100)) 
lscore = np.zeros((len(prep), len(cats), len(cats))) 
for icon, concepti in enumerate(prep):
    print(concepti)
    for icat, categoryi in enumerate(cats):
        for jcat, categoryj in enumerate(cats):
        print(categoryi)
        print(categoryj)
        Xi = lfc8[icon, icat, 0:nb_img]
        Xj = lfc8[icon, jcat, 0:nb_img]
        X = np.vstack([Xi, Xj])
        y = np.hstack((np.ones((nb_img., )), np.zeros((nb_img,)))) 
        # true scores
        skf = StratifiedKFold(y, 5)
        true_score = []
        for train, test in skf:
             # train a svm on the output
            clf = svm.SVC()
            clf.fit(X[train], y[train])
            true_score.append(clf.score(X[test], y[test]))
        print(np.mean(true_score))
        lscore[icon, icat, jcat] = np.mean(true_score)
        for i in range(100):
            np.random.shuffle(y)
            skf = StratifiedKFold(y, 5)
            ran_score = []
            for train, test in skf:
                 # train a svm on the output
                clf = svm.SVC()
                clf.fit(X[train], y[train])
                ran_score.append(clf.score(X[test], y[test]))
            lran[icon, icat, jcat, i] = np.mean(ran_score)
np.save(os.path.join("../../data/res/", "all_across_lran.npy"), lran)
np.save(os.path.join("../../data/res/", "all_across_score.npy"), lscore)


