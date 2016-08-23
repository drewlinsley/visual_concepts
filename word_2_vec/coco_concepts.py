import os
import numpy as np

path = "../../data/"

def load_dic(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        return pickle.load(f)

emb = np.load(os.path.join(path, "embeddings.npy"))
prep = np.loadtxt('../prob_word_lists/prepositions.csv', type('str'))
dic = load_dic(path, 'dictionary')
rdic = load_dic(path, 'reverse_dictionary')

word_index = []
for word in prep:
    word_index.append(dic[word])
    
# find the corresponding vector in the embedding space
emb[word_index]

