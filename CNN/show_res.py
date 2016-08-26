import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

path_dat = "../../data/"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
prep_len = len(prep)
cats = os.listdir(os.path.join(path_dat, "imgs"))

lran = np.load(os.path.join("../../data/res/", "lran.npy"))
lscore = np.load(os.path.join("../../data/res/", "lscore.npy"))

ran = lran.reshape((prep_len, -1, 100))
score = lscore.reshape((prep_len, -1))
pval = np.round((np.sum(score[:, :, np.newaxis]>ran, axis=2)+1)/(ran.shape[2]+1.), 3)

# swarmplots
for icat, cat in enumerate(cats):
    plt.figure()
    sns.set_style("whitegrid")
    ax2 = sns.swarmplot(data=ran[:, icat].T, zorder=1)
    ax1 = plt.scatter(range(score[:,icat].shape[0]), score[:, icat], color='r', zorder=2)
    plt.xticks(range(len(prep)), prep, rotation='40')
    plt.title(cat)
    for icon, con in enumerate(prep):
        plt.text(icon-0.3, ax2.get_ylim()[1]-0.02, pval[icon, icat], rotation=30)
    plt.savefig('../../data/fig/'+ cat + 'swarmplot.png')
    
# in bears, comparing concepts
score = np.load(os.path.join("../../data/res/", "across_score.npy"))
ran = np.load(os.path.join("../../data/res/", "across_lran.npy"))
#swarmplots
#ran = np.random.rand((12, 12, 100))
fig = plt.figure(figsize=(20, 20))
for icon, concepti in enumerate(prep):
    fig.text(0.07, 1-(0.12+icon/16.), concepti, rotation=30) 
    fig.text(0.12+icon/16., 0.94, concepti, rotation=30) 
    print(icon)
    for jcon, conceptj in enumerate(prep[icon:]):
        jcon += icon
        print(jcon)
        fig.add_subplot(13, 13, 1 + 13*icon +  jcon)
        ax = sns.swarmplot(x=ran[icon, jcon], zorder=1)
        ax.scatter(score[icon,jcon], 0., color='r', zorder=2)
plt.savefig('../../data/fig/'+ 'swarm_across_plot.png')

# matrix
score = (score -0.48)
#score -= np.diag(np.diag(score))
# Generate a mask for the upper triangle
mask = np.zeros_like(score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure()
sns.heatmap(score, mask=mask, cmap=cmap)
plt.savefig('../../data/fig/'+ 'matrix_across_plot.png')

# using panda dataframe
sc = pd.DataFrame(score)
sc.columns = cats 


"""
plt.figure()
plt.violinplot(lres, range(len(prep)), showmedians=True)
plt.xticks(range(len(prep)), prep, rotation='40')
plt.savefig('../../data/violinplot.png')
plt.figure()
plt.boxplot(lres, range(len(prep)))
plt.xticks(range(len(prep)), prep, rotation='40')
plt.savefig('../../data/boxplot.png')
"""
"""
Maybe you could train on one animal's concepts and test on another's
Something else to try: look online for something that will give you the low-level features of these images. One possibility is the gist package from oliva  and torrialba. Use that in place of the cnn to establish a "low level visual property" baseline
"""
