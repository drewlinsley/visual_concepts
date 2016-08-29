import numpy as np
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
import os, copy

path_dat = "../../data"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = np.loadtxt(os.path.join(path_dat, "categories.txt"), type('str'))
lfc8 = np.load(os.path.join(path_dat, 'output_vgg19.npy')).reshape((len(prep), len(cats), 100, 1000))
labels = np.load(os.path.join(path_dat, 'labels_vgg19.npy')).reshape((len(prep), len(cats), 100))
nb_img = 100 

lran, lscore = [], [] 
for icon, concept in enumerate(prep):
    print(concept)
    X = lfc8[icon].reshape(len(cats)*100, 1000)
    y = np.tile(np.hstack((np.ones((nb_img/2, )), np.zeros((nb_img/2,)))), len(cats)) 
    # true scores
    skf = StratifiedKFold(y, 5)
    for train, test in skf:
         # train a svm on the output
        clf = svm.SVC()
        clf.fit(X[train], y[train])
        lscore.append(clf.score(X[test], y[test]))
        ytest = copy.deepcopy(y)[test] # because we shuffle afterwards
        for i in range(100):
            np.random.shuffle(ytest)
            lran.append(clf.score(X[test], ytest))

lran = np.array(lran).reshape((len(prep), -1, 100))
lscore = np.array(lscore).reshape((len(prep), -1))
ran = np.mean(lran, 1)
score = np.mean(lscore, 1)

np.save(os.path.join("../../data/res/", "across_all_cat_ran.npy"), ran)
np.save(os.path.join("../../data/res/", "across_all_cat_score.npy"), score)


