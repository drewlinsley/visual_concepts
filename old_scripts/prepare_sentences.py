#Run wang2vec
#Then prepare_sentences
#Followed by word_script_v2
#And finally im_mosaic

import numpy as np
import scipy.spatial.distance as dis
import csv

def read_file(name, deli):
	with open(name,'r') as f:
		reader=csv.reader(f,delimiter=deli,quoting=csv.QUOTE_NONE)
		out = list(reader)
	return out

def split_data(d):
	#Loop through, splitting off the first cell of each vector and adding it to words
	words = [];
	data = [];
	for cells in d:
		words.append(cells[0])
		data.append(map(float,cells[1:-1]))
	data = np.asarray(data)
	return words, data

#Reads sentences and Word2vec vector space. Accumulates the average vector for words in each sentence. These are used to represent sentences.
#d = read_file('caps.txt',' ') #Load word2vec data
#d = read_file('testvec.txt',' ') #Load word2vec data
d = read_file('full_wiki_wangvec.txt',' ') #Load word2vec data
words, data = split_data(d);
my_captions = np.load('captions_train2014.npy')
for m in range(0,len(my_captions)):
	my_captions[m] = [x.encode('ascii') for x in my_captions[m]]
num_dims = data.shape[1]

#Get average W2V for each sentence
mu_word = np.mean(data,axis=0)
sentences = np.zeros((my_captions.shape[0],data.shape[1]))
num_captions = my_captions.shape[0]
for idx in range(0,num_captions):
	it_sentence = my_captions[idx]
	it_sentence_len = len(it_sentence)
	temp_sentence = np.empty((it_sentence_len,data.shape[1]))
	temp_sentence[:] = np.NAN
	for il in range(0,it_sentence_len):
		it_word = it_sentence[il]
		if it_word in words:
			ti = words.index(it_word)
			temp_sentence[il,:] = data[ti,:]
	if np.sum(np.isnan(temp_sentence)) == it_sentence_len * num_dims: #if all are nan
		sentences[idx,:] = mu_word
	else:
		#sentences[idx,:] = np.divide(np.nanmean(temp_sentence,axis=0),np.nanstd(temp_sentence + 1e-4,axis=0))
		sentences[idx,:] = np.nanmean(temp_sentence,axis=0)
		#sentences[idx,:] = np.nanmax(temp_sentence,axis=0)

file_name = 'sentences.npy'
np.save(file_name,sentences)