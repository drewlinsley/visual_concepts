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


## swarmplots for each category and each concept
lran = np.load(os.path.join("../../data/res/", "lran.npy"))
lscore = np.load(os.path.join("../../data/res/", "lscore.npy"))
ran = lran.reshape((prep_len, -1, 100))
score = lscore.reshape((prep_len, -1))
pval = np.round((np.sum(score[:, :, np.newaxis]>ran, axis=2)+1)/(ran.shape[2]+1.), 3)
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
    
## compare concepts
score = np.load(os.path.join("../../data/res/", "across_score.npy"))
ran = np.load(os.path.join("../../data/res/", "across_lran.npy"))
# matrix of swarmplots
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
# imshow of the values
score = (score -0.48)
#score -= np.diag(np.diag(score))
# Generate a mask for the upper triangle
mask = np.zeros_like(score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure()
sns.heatmap(score, mask=mask, cmap=cmap)
plt.savefig('../../data/fig/'+ 'matrix_across_plot.png')

## swarmplots across categories
lran = np.load(os.path.join("../../data/res/", "across_cat_lran.npy"))
lscore = np.load(os.path.join("../../data/res/", "across_cat_score.npy"))
plt.figure()
sns.set_style("whitegrid")
ax2 = sns.swarmplot(data=lran.T, zorder=1)
ax1 = plt.scatter(range(lscore.shape[0]), lscore, color='r', zorder=2)
plt.xticks(range(len(prep)), prep, rotation='40')
plt.title('across categories')
for icon, con in enumerate(prep):
    plt.text(icon-0.3, ax2.get_ylim()[1]-0.02, pval[icon, icat], rotation=30)
plt.savefig('../../data/fig/across_categories_swarmplot.png')

## swarmplots for two categories
lran = np.load(os.path.join("../../data/res/", "two_cat_lran.npy"))
lscore = np.load(os.path.join("../../data/res/", "two_cat_score.npy"))
plt.figure()
sns.set_style("whitegrid")
ax2 = sns.swarmplot(data=lran.T, zorder=1)
ax1 = plt.scatter(range(lscore.shape[0]), lscore, color='r', zorder=2)
plt.xticks(range(len(prep)), prep, rotation='40')
plt.title('Two categories: a and b')
for icon, con in enumerate(prep):
    plt.text(icon-0.3, ax2.get_ylima
plt.savefig('../../data/fig/two_categories_swarmplot.png')

# multiple bivariate kde all
ran = np.load(os.path.join("../../data/res/", "all_across_lran.npy"))
score = np.load(os.path.join("../../data/res/", "all_across_score.npy"))
pval = np.round((np.sum(score[:, :, :, np.newaxis]>ran, axis=3)+1)/(ran.shape[3]+1.), 3)
plt.figure()
color = ["Reds", "Blues", "Greens", "Oranges", "deep", "muted", "bright", "pastel", "dark", "colorblind", "hls", "husl"]
for icon, concept in enumerate(prep):
    ax = sns.kdeplot(np.flatten(score[icon]), np.flatten(pval[icon]),
                      cmap=color[icon], shade=True, shade_lowest=False)
    ax1 = plt.scatter(np.diag(score[icon]), np.diag(pval[icon]), color='r', zorder=2)
    col = sns.color_palette(color[icon])[-2]
    ax.text(2.5, 8.2, concept, size=16, color=col)
plt.title('KDE of all the concepts')
plt.savefig('../../data/fig/all_concepts.png')


 
"""
# using panda dataframe
sc = pd.DataFrame(score)
sc.columns = cats 


plt.figure()
plt.violinplot(lres, range(len(prep)), showmedians=True)
plt.xticks(range(len(prep)), prep, rotation='40')
plt.savefig('../../data/violinplot.png')
plt.figure()
plt.boxplot(lres, range(len(prep)))
plt.xticks(range(len(prep)), prep, rotation='40')
plt.savefig('../../data/boxplot.png')
"""
