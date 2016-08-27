import os, sys, pickle, shutil
import numpy as np
sys.path.append('../../coco/PythonAPI')
from pycocotools.coco import COCO
import string

path = "../../data/"
num_ex = 5
train_type = 'cbow_targeted_sampling'

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        print('directory already exists')


# Load embeddings
def load_dic(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        return pickle.load(f)

if train_type=='skip_gram':
    emb = np.load(os.path.join(path, "embeddings.npy"))
    dic = load_dic(path, 'dictionary')
    rdic = load_dic(path, 'reverse_dictionary')
elif train_type=='cbow':
    emb = np.load(os.path.join(path, "embeddings_cbow.npy"))
    dic = load_dic(path, 'dictionary_cbow')
    rdic = load_dic(path, 'reverse_dictionary_cbow')
elif train_type=='cbow2':
    emb = np.load(os.path.join(path, "embeddings_cbow2.npy"))
    dic = load_dic(path, 'dictionary_cbow2')
    rdic = load_dic(path, 'reverse_dictionary_cbow2')
elif train_type=='cbow_targeted_sampling':
    emb = np.load(os.path.join(path, "embeddings_cbow_targeted_sampling.npy"))
    dic = load_dic(path, 'dictionary_cbow_targeted_sampling')
    rdic = load_dic(path, 'reverse_dictionary_cbow_targeted_sampling')



# Find embedding for chosen concepts
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))

word_index = []
for word in prep:
    word_index.append(dic[word])
emb_cpt = emb[word_index]

# Find average embedding for each caption of COCO images in each category
coco_caps = COCO(os.path.join(path, 'coco', 'annotations', 'captions_train2014.json'))
coco_anns = COCO(os.path.join(path, 'coco', 'annotations', 'instances_train2014.json'))

lcats = ['bear', 'bird', 'cat', 'cow', 'giraffe', 'elephant', 'dog', 'horse',
         'mouse', 'sheep', 'zebra', 'wine glass', 'vase', 'TV', 'toaster', 
         'suitcase', 'scissors', 'potted plant', 'microwave', 'laptop', 'hand bag',
         'hairdryer', 'cup', 'couch', 'chair', 'bottle', 'book', 'backpack'] 

cats = coco_anns.getCatIds(catNms=lcats)
cats_name = coco_anns.loadCats(cats)

for cat in cats:
    print(cat)
    imIds = coco_anns.getImgIds(catIds=cat)
    emb_cap, limIds, lannIds = [], [], []
    for imId in imIds:
        annIds = coco_caps.getAnnIds(imgIds=imId)
        lannIds.extend(annIds)
        anns = coco_caps.loadAnns(annIds)
        # taking the mean of each caption in the embedding space and stacking it into an array
        for iann, ann in enumerate(anns):
            lann = str(ann['caption']).translate(string.maketrans("",""), string.punctuation).split(' ')
            lind = []
            for wi in lann:
                try:
                    lind.append(dic[wi.lower().replace(" ","").replace("\n","")])
                except KeyError:
                    #print(wi.lower() + ' not in dictionary')
                    pass
            if lind==[]:raise ValueError('list empty because no words of the caption is in the dictionary')
            emb_cap.append(np.mean(emb[lind], axis=0))
            limIds.append(imId)
            

    emb_cap = np.array(emb_cap)
    # compute the cosine distance between concepts and captions
    emb_cap_norm = np.divide(emb_cap, np.sqrt(np.sum(np.square(emb_cap), axis=1)[:, np.newaxis]))
    emb_cpt_norm = np.divide(emb_cpt, np.sqrt(np.sum(np.square(emb_cpt), axis=1)[:, np.newaxis]))
    similarity = np.dot(emb_cap_norm, emb_cpt_norm.T)

    path_cat = os.path.join(path, 'imgs', coco_anns.loadCats(cat)[0]['name'])
    for ind in range(similarity.shape[1]):
        ind_best = np.argsort(similarity[:, ind])[-num_ex:] 
        ind_worst = np.argsort(similarity[:, ind])[:num_ex] 
        best_caps = coco_caps.loadAnns(list(np.array(lannIds)[ind_best]))
        worst_caps = coco_caps.loadAnns(list(np.array(lannIds)[ind_worst]))
        best_imgs = coco_caps.loadImgs(list(np.array(limIds)[ind_best]))
        worst_imgs = coco_caps.loadImgs(list(np.array(limIds)[ind_worst]))
        path_pos = os.path.join(path_cat, prep[ind], 'positive')
        path_neg = os.path.join(path_cat, prep[ind], 'negative')
        make_dir(path_pos)
        make_dir(path_neg)
        with open(os.path.join(path_cat, prep[ind], 'captions_positive_'+prep[ind]+'.txt'), 'wb') as f:
            for cap in reversed(best_caps):
                f.write(cap['caption'])
                f.write('\n')
        with open(os.path.join(path_cat, prep[ind], 'captions_negative_'+prep[ind]+'.txt'), 'wb') as f:
            for cap in worst_caps:
                f.write(cap['caption'])
                f.write('\n')
        for iimg, img in enumerate(reversed(best_imgs)):
            shutil.copyfile(os.path.join(path, 'coco', 'train2014', img['file_name']), os.path.join(path_pos, str(iimg) + '.jpg'))
        for iimg, img in enumerate(worst_imgs):
            shutil.copyfile(os.path.join(path, 'coco', 'train2014', img['file_name']), os.path.join(path_neg, str(iimg) + '.jpg'))
