import numpy as np
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
import os, copy

path_dat = "../../data"
lfc8 = np.load(os.path.join(path_dat, 'output_vgg19.npy')).reshape(-1, 1000)
labels = np.load(os.path.join(path_dat, 'labels_vgg19.npy'))
nb_img = 50 

prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = os.listdir(os.path.join(path_dat, "imgs"))

lran = np.zeros((len(prep), len(prep), 100)) 
lscore = np.zeros((len(prep), len(prep))) 
for icon, concepti in enumerate(prep):
    for jcon, conceptj in enumerate(prep):
        print(concepti)
        print(conceptj)
        icat = 8
        category = cats[icat]
        print(category)
        Xi = lfc8[icon*(len(cats)*nb_img)+icat*nb_img:(icon)*(len(cats)*nb_img)+(icat+1)*nb_img]
        Xj = lfc8[jcon*(len(cats)*nb_img)+icat*nb_img:(jcon)*(len(cats)*nb_img)+(icat+1)*nb_img]
        X = np.vstack([Xi, Xj])
        y = np.hstack((np.ones((nb_img, )), np.zeros((nb_img,)))) 
        # true scores
        skf = StratifiedKFold(y, 5)
        true_score = []
        for train, test in skf:
             # train a svm on the output
            clf = svm.SVC()
            clf.fit(X[train], y[train])
            true_score.append(clf.score(X[test], y[test]))
        print(np.mean(true_score))
        lscore[icon, jcon] = np.mean(true_score)
        for i in range(100):
            np.random.shuffle(y)
            skf = StratifiedKFold(y, 5)
            ran_score = []
            for train, test in skf:
                 # train a svm on the output
                clf = svm.SVC()
                clf.fit(X[train], y[train])
                ran_score.append(clf.score(X[test], y[test]))
            lran[icon, jcon, i] = np.mean(ran_score)
np.save(os.path.join("../../data/res/", "across_lran.npy"), lran)
np.save(os.path.join("../../data/res/", "across_score.npy"), lscore)


