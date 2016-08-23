import os, sys
import numpy as np
sys.path.append('../../coco/PythonAPI')
from pycocotools.coco import COCO

path = "../../data/"

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

# Find average embedding for each caption of COCO images
coco_caps = COCO(os.path.join(path, 'coco', 'annotations', 'captions_train2014.json'))
annIds = coco_caps.getAnnIds()
anns = coco_caps.loadAnns(annIds)

emb_cap = []
for ann in anns[0:50]:
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

similarity = np.dot(emb_cap_norm, emb_cpt.T)

best_con = prep[np.argmax(similarity, axis=1)]

np.savetxt(os.path.join(path, 'best_concepts.txt'), best_con.astype('str'), fmt='%s')
