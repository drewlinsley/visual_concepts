import numpy as np
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
import os, copy

path_dat = "../../data"
lfc8 = np.load(os.path.join(path_dat, 'output_vgg19.npy')).reshape(-1, 1000)
labels = np.load(os.path.join(path_dat, 'labels_vgg19.npy'))
nb_img = 100 

prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = np.loadtxt(os.path.join(path_dat, "categories.txt"), type('str'))

lran = []
lscore = []
for icon, concept in enumerate(prep):
    print(concept)
    for icat, category in enumerate(cats):
        print(category)
        # randomization
        X = lfc8[icon*(len(cats)*nb_img)+icat*nb_img:(icon)*(len(cats)*nb_img)+(icat+1)*nb_img]
        y = copy.deepcopy(labels[icon*(len(cats)*nb_img)+icat*nb_img:(icon)*(len(cats)*nb_img)+(icat+1)*nb_img])
        # true scores
        skf = StratifiedKFold(y, 5)
        true_score = []
        for train, test in skf:
             # train a svm on the output
            clf = svm.SVC()
            clf.fit(X[train], y[train])
            true_score.append(clf.score(X[test], y[test]))
        print(np.mean(true_score))
        lscore.append(np.mean(true_score))
        for i in range(X.shape[0]):
            np.random.shuffle(y)
            skf = StratifiedKFold(y, 5)
            ran_score = []
            for train, test in skf:
                 # train a svm on the output
                clf = svm.SVC()
                clf.fit(X[train], y[train])
                ran_score.append(clf.score(X[test], y[test]))
            lran.append(np.mean(ran_score))
        print np.round(lran[-X.shape[0]:], 2)
        np.save(os.path.join("../../data/res/", "lran.npy"), lran)
        np.save(os.path.join("../../data/res/", "lscore.npy"), lscore)

lran = np.array(lran)
lscore = np.array(lscore)
np.save(os.path.join("../../data/res/", "lran.npy"), lran)
np.save(os.path.join("../../data/res/", "lscore.npy"), lscore)


