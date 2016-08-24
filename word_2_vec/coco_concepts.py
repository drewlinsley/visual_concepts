import os, sys, shutil
import numpy as np
sys.path.append('../../coco/PythonAPI')
from pycocotools.coco import COCO

path = "../../data/"
num_ex = 50

# Load embeddings
def load_dic(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        return pickle.load(f)

emb = np.load(os.path.join(path, "embeddings.npy"))
dic = load_dic(path, 'dictionary')
rdic = load_dic(path, 'reverse_dictionary')

# Find embedding for chosen concepts
prep = np.loadtxt('../prob_word_lists/prepositions.csv', type('str'))
word_index = []
for word in prep:
    word_index.append(dic[word])
emb_cpt = emb[word_index]

# Find average embedding for each caption of COCO images in each category
coco_caps = COCO(os.path.join(path, 'coco', 'annotations', 'captions_train2014.json'))
cats = coco_caps.loadCats(coco_caps.getCatIds())
for cat in cats:
    annIds = coco_caps.getAnnIds(catIds=cat)
    anns = coco_caps.loadAnns(annIds)
    # taking the mean of each caption in the embedding space and  stacking in an array
    emb_cap = []
    for ann in anns:
        lann = ann['caption'].split(' ')
        lind = []
        for wi in lann:
            try:
                lind.append(dic[wi])
            except KeyError:
                print(wi + ' not in dictionary')
        emb_cap.append(np.mean(emb[lind], axis=0))
    emb_cap = np.array(emb_cap)
    # compute the cosine distance between concepts and captions
    emb_cap_norm = np.divide(emb_cap, np.sqrt(np.sum(np.square(emb_cap), axis=1))[:, np.newaxis])
    emb_cpt_norm = np.divide(emb_cpt, np.sqrt(np.sum(np.square(emb_cpt), axis=1)[:, np.newaxis]))
    similarity = np.dot(emb_cap_norm, emb_cpt_norm.T)
    os.mkdir(os.path.join(path, cat))
    for ind in range(similarity.shape[1]):
        ind_best = np.argsort(np.find(np.argmax(similarity, axis=1)==ind))
        best_con = prep[ind_best]
        best_caps = anns[ind_best]
        os.mkdir(os.path.join(path, ind))
        np.savetxt(os.path.join(path, cat, ind, 'best_concepts_'+prep[ind]+'.txt'), best_con.astype('str'), fmt='%s')
        np.savetxt(os.path.join(path, cat, ind, 'best_captions_'+prep[ind]+'.txt'), best_con.astype('str'), fmt='%s')
        img = coco_caps.loadImgs(anns[ind_best]['image_id'])[0]
        shutil.copyfile(os.path.join(path, 'coco', 'train2014', img['file_name'], os.path.join(path, cat, ind, img['file_name'])
