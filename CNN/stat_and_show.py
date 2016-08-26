import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

path_dat = "../../data/"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
lres = []
for concept in prep:
    cres = []
    #for category in os.listdir(os.path.join(path_dat, "imgs"):
    lcats = ['bear', 'bird', 'cat', 'cow', 'giraffe', 'elephant', 'dog', 'horse',
         'mouse', 'sheep', 'suitcase', 'scissors', 'microwave', 'laptop',
         'cup', 'couch', 'chair', 'bottle', 'book', 'backpack'] 
    for category in lcats:
        cres.extend(np.load(os.path.join(path_dat, 'res', category + '_' + concept + '.npy')))
    lres.append(cres)

plt.figure()
plt.violinplot(lres, range(len(prep)), showmedians=True)
plt.xticks(range(len(prep)), prep, rotation='40')
plt.savefig('../../data/violinplot.png')
plt.figure()
plt.boxplot(lres, range(len(prep)))
plt.xticks(range(len(prep)), prep, rotation='40')
plt.savefig('../../data/boxplot.png')
