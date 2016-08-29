import numpy as np
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
import os, copy

"""
training on one category and testing on another one, for all possible pairs of category
"""

path_dat = "../../data"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = np.loadtxt(os.path.join(path_dat, "categories.txt"), type('str'))
lfc8 = np.load(os.path.join(path_dat, 'output_vgg19.npy')).reshape(len(prep), len(cats), 100, 1000)
labels = np.load(os.path.join(path_dat, 'labels_vgg19.npy')).reshape(len(prep), len(cats), 100)
nb_img = 100 

# fewer categories 
lran, lscore = [], [] 
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
                lscore.append(clf.score(Xtest[test], ytest[test]))
                ytestran = copy.deepcopy(ytest)[test] 
                for i in range(Xtrain.shape[0]):
                    np.random.shuffle(ytestran)
                    lran.append(clf.score(Xtest[test], ytestran))

lran = np.array(lran).reshape((len(prep), len(cats), len(cats), -1, Xtrain.shape[0]))
lscore = np.array(lscore).reshape((len(prep), len(cats), len(cats), -1))
ran = np.mean(lran, 3)
score = np.mean(lscore, 3)
np.save(os.path.join("../../data/res/", "train_test_different_categories_all_ran.npy"), ran)
np.save(os.path.join("../../data/res/", "train_test_different_categories_all_score.npy"), score)


