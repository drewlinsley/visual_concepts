import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
%autoindent

path_dat = "../../data/"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
prep_len = len(prep)
cats = np.loadtxt(os.path.join(path_dat, "categories.txt"), type('str'))

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
cat = 2
score = np.load(os.path.join("../../data/res/", cats[cat]+"_across_score.npy"))
ran = np.load(os.path.join("../../data/res/", cats[cat]+"_across_lran.npy"))
# matrix of swarmplots
fig = plt.figure(figsize=(20, 20))
for icon, concepti in enumerate(prep):
    fig.text(0.09+icon/13., 1-(0.12+icon/12.), concepti, rotation=30) 
    fig.text(0.15+icon/13., 0.93, concepti, rotation=30) 
    print(icon)
    for jcon, conceptj in enumerate(prep[icon:]):
        jcon += icon
        print(jcon)
        fig.add_subplot(10, 10, 1 + 10*icon +  jcon)
        ax = sns.swarmplot(x=ran[icon, jcon], zorder=1)
        ax.scatter(score[icon,jcon], 0., color='r', zorder=2)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['.2', '.4', '.6', '.8', '1' ])
fig.text(0.35, 0.35, cats[cat], fontsize='40')
plt.savefig('../../data/fig/'+ cats[cat]+'_swarm_across_concepts_plot.png')
# imshow of the values
score = (score -0.48)
#score -= np.diag(np.diag(score))
# Generate a mask for the upper triangle
mask = np.zeros_like(score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

cat = 0
fig = plt.figure()
fig.text(0.5, 0.55, cats[cat], fontsize='20')
ax = sns.heatmap(score, mask=mask, cmap=cmap)
ax.set_xticklabels(prep, rotation=30)
ax.set_yticklabels(prep[::-1], rotation=30)
plt.savefig('../../data/fig/'+ cats[cat]+'_matrix_across_concepts_plot.png')

## swarmplots across categories
lran = np.load(os.path.join("../../data/res/", "across_all_cat_lran.npy"))
lscore = np.load(os.path.join("../../data/res/", "across_all_cat_score.npy"))
pval = np.round((np.sum(lscore[:, np.newaxis]>lran, axis=1)+1)/(lran.shape[1]+1.), 3)
plt.figure()
sns.set_style("whitegrid")
ax2 = sns.swarmplot(data=lran.T, zorder=1)
ax1 = plt.scatter(range(lscore.shape[0]), lscore, color='r', zorder=2)
plt.xticks(range(len(prep)), prep, rotation='40')
plt.title('across categories')
for icon, con in enumerate(prep):
    plt.text(icon-0.3, ax2.get_ylim()[1]-0.02, pval[icon], rotation=30)
plt.savefig('../../data/fig/across_all_categories_swarmplot.png')

## swarmplots for two categories
cat1 = 1
cat2 = 2
ran = np.load(os.path.join("../../data/res/", "all_across_lran.npy"))
score = np.load(os.path.join("../../data/res/", "all_across_score.npy"))
pval = np.round((np.sum(score[:, :, :, np.newaxis]>ran, axis=3)+1)/(ran.shape[3]+1.), 3)
plt.figure()
sns.set_style("whitegrid")
ax2 = sns.swarmplot(data=ran[:, cat1, cat2].T, zorder=1)
ax1 = plt.scatter(range(score.shape[0]), score[:, cat1, cat2], color='r', zorder=2)
plt.xticks(range(len(prep)), prep, rotation='40')
plt.title('Two categories: '+  cats[cat1] + ' and ' + cats[cat2])
for icon, con in enumerate(prep):
    plt.text(icon-0.3, ax2.get_ylim()[1]-0.02, pval[icon, 0, 1], rotation=30)
plt.savefig('../../data/fig/'+cats[cat1]+'_'+cats[cat2]+'_swarmplot.png')

# across all concepts 
ran = np.load(os.path.join("../../data/res/", "all_across_lran.npy"))
score = np.load(os.path.join("../../data/res/", "all_across_score.npy"))
pval = np.round((np.sum(score[:, :, :, np.newaxis]>ran, axis=3)+1)/(ran.shape[3]+1.), 3)
plt.figure()
_, ax1 = plt.subplots(1)
ax = sns.swarmplot(data=score.reshape(-1, 9).T, zorder=1)
ax1.scatter(range(score.shape[0]), score[:, 0, 0], color='r', alpha=0.8, zorder=2)
ax1.scatter(range(score.shape[0]), score[:, 1, 1], color='b', alpha=0.8, zorder=2)
ax1.scatter(range(score.shape[0]), score[:, 2, 2], color='g', alpha=0.8, zorder=2)
ax1.legend(['Bear', 'Elephant', 'Horse'])
lgd = ax1.get_legend()
lgd.legendHandles[0].set_color(plt.cm.Reds(.8))
lgd.legendHandles[1].set_color(plt.cm.Blues(.8))
lgd.legendHandles[2].set_color(plt.cm.Greens(.8))
plt.xticks(range(len(prep)), prep, rotation='40')
plt.title('Concepts for all the categories')
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
