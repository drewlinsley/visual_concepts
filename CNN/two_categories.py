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
cats = os.listdir(os.path.join(path_dat, "imgs"))

lran = np.zeros((len(prep), 100)) 
lscore = np.zeros((len(prep), )) 
for icon, concept in enumerate(prep):
    print(concept)
    cat1 = 0 
    print(cats[cat1])
    cat2 = 2 
    print(cats[cat2])
    Xtrain = lfc8[icon*(len(cats)*nb_img)+cat1*nb_img:(icon)*(len(cats)*nb_img)+(cat1+1)*nb_img]
    Xtest = lfc8[icon*(len(cats)*nb_img)+cat2*nb_img:(icon)*(len(cats)*nb_img)+(cat2+1)*nb_img]
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
    lscore[icon] = np.mean(true_score)
    for i in range(100):
        np.random.shuffle(ytrain)
        skf = StratifiedKFold(ytrain, 5)
        ran_score = []
        for train, test in skf:
             # train a svm on the output
            clf = svm.SVC()
            clf.fit(Xtrain[train], ytrain[train])
            ran_score.append(clf.score(Xtest[test], ytest[test]))
        lran[icon, i] = np.mean(ran_score)
np.save(os.path.join("../../data/res/", "b_h_cat_lran.npy"), lran)
np.save(os.path.join("../../data/res/", "b_h_cat_score.npy"), lscore)


