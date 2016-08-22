from __future__ import division, print_function

import numpy as np
from collections import Counter
#import dvisword2vec
import csv
import scipy.spatial.distance as dis
import re
import operator
from nltk import pos_tag_sents
import codecs
import json 

def read_file(name, deli):
	with open(name,'r') as f:
		reader=csv.reader(f,delimiter=deli,quoting=csv.QUOTE_NONE)
		out = list(reader)
	return out

def read_json(name):
	with open(name,) as f:
		out=json.load(f)
	return out

def prepare_jsons(train,test,field):
	coco_train_md = read_json(train) #Load coco meta data
	coco_val_md = read_json(test) #Load coco meta data
	coco_train_images = coco_train_md[field]
	coco_val_images = coco_val_md[field]
	coco_all_images = coco_train_images + coco_val_images
	return coco_all_images

def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

def split_data(d):
	#Loop through, splitting off the first cell of each vector and adding it to words
	words = [];
	data = [];
	for cells in d:
		words.append(cells[0])
		data.append(map(float,cells[1:-1]))
	data = np.asarray(data)
	return words, data

def fix_words(words):
	#Remove nonletter characters from words
	regex = re.compile('[^a-zA-Z]')
	new_words = []
	for nw in words:
		new_words.append(regex.sub('',nw).lower())
	return new_words

def preposition_count(new_words):
	prep_idx = []
	prep_count = []
	for idx in prepositions:
		prep_idx.append(index_containing_substring(new_words,idx[0]))
		prep_count.append(new_words.count(idx[0]))
		#For some reason some of these appear multiple times... multiple word2vec "contexts" I guess?
	prep_idx = np.asarray(prep_idx)
	prep_count = np.asarray(prep_count)
	prep_mask = prep_idx != -1;
	prep_idx = prep_idx[prep_mask];
	prep_count = prep_count[prep_mask];
	return prep_idx, prep_count

def find_nearest_sentences(caption_data,prep_count,prep_idx,data,coco_all_images,new_words,coco_all_captions,side):
	#2. For each preposition, get the N most/least similar nouns
	#a. precalculate distance between each preposition and all other words
	if side == 'most':
		side = -1
		print('Using side ' + str(side))
	else:
		side = 1
		print('Using side ' + str(side))

	#dm = np.empty((prep_count.shape[0],caption_data.shape[0]))
	dm = np.empty((1,caption_data.shape[0]))
	preposition_word_matches = []
	preposition_word_images = []
	preposition_word_image_pointers = []
	preposition_word_alt_image_pointers = []
	preposition_ids = []
	word_pos = []
	prep_list = []
	n = 50;
	m = 200;
	total_props = dm.shape[0]
	#Evaluate argmax[1...N](sentences|prep) -- find the N most likely sentences given each preposition.
	for idx in range(0,total_props):
		print('******Gathering nearest neighbors for preposition ' + str(idx) + '/' + str(total_props) + ': ' + new_words[prep_idx[idx]])
		prep_list.append(new_words[prep_idx[idx]])
		for il in range(0,caption_data.shape[0]):
			#dm[idx,il] = dis.cosine(data[prep_idx[idx]],caption_data[il])
			dm[il] = dis.cosine(data[prep_idx[idx]],caption_data[il])
		#1. For each preposition, find the POS of the nearest and furthest (Catch trials) m words 
		nearest_ids = np.argsort(dm[0,::side])[:m]

		#b. Preallocate POS tagging
		pos_list = [];
		for il in range(0,m):
			twrd = new_words[nearest_ids[il]]
			if twrd == '':
				twrd = 'over' #if empty, set to a preposition so we don't keep it.
			pos_list.append([twrd])
		pos_list = pos_tag_sents(pos_list)

		#c. Preallocate variables for assignment loop
		i = 0
		it_word_matches = []
		it_images = []
		it_image_pointers = []
		it_alt_image_pointers = []
		it_word_pos = []
		it_ids = []
		while i < n:
			curr_word = new_words[nearest_ids[i]]
			if curr_word == '':
				print('tossing ' + curr_word + ' (empty)')
			else:
				pos = pos_list[i][0]
				if pos[1] in ('NN','NNS'): #Target nouns for now. Can work on verbs later.
					it_word_matches.append(curr_word) #Keep this word
					#Also store the pointer to its file name
					it_images.append(coco_all_images[nearest_ids[i]][u'file_name'])
					#And web address
					it_image_pointers.append(coco_all_images[nearest_ids[i]][u'flickr_url'])
					#And alt web address
					it_alt_image_pointers.append(coco_all_images[nearest_ids[i]][u'coco_url'])
					#Also store the POS
					it_word_pos.append(pos[1])
					#Also store the id
					#it_ids.append(coco_all_images[nearest_ids[i]][u'id'])
					it_ids.append(coco_all_captions[nearest_ids[i]][u'caption'])
					#And iterate
					print('keeping ' + curr_word + ' = ' + pos[1])
				else:
					print('tossing ' + curr_word + ' = ' + pos[1])
			i += 1
		preposition_word_matches.append(it_word_matches)
		preposition_word_images.append(it_images)
		preposition_word_image_pointers.append(it_image_pointers)
		preposition_word_alt_image_pointers.append(it_alt_image_pointers)
		word_pos.append(it_word_pos)
		preposition_ids.append(it_ids)
	return dm, preposition_word_matches, preposition_word_images, preposition_word_image_pointers, preposition_word_alt_image_pointers, word_pos, prep_list, preposition_ids

def save_json(input,path):
	json.dump(input, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

def prepare_json_files(to_save,names):
	for i in range(0,len(to_save)):
		path = names[i] + ".json"
		save_json(to_save[i],path)


#./word2vec -train ../../Downloads/captions_train2014.txt -output caps.txt -size=200 -window=3 -type=2 -iter=30 -binary=0


#Get down to business
d = read_file('caps.txt',' ') #Load word2vec data
prepositions = read_file('prepositions.csv',',') #Load prepositions
coco_all_images = prepare_jsons('instances_train2014.json','instances_val2014.json',u'images')

#Load in captions, too
#coco_all_captions = prepare_jsons('annotations/captions_train2014.json','annotations/captions_val2014.json',u'annotations')
my_captions = np.load('captions_train2014.npy') #each sentence used to train the word model
caption_data = np.load('sentences.npy')

#Loop through, splitting off the first cell of each vector and adding it to words
words, data = split_data(d);

#Remove nonletter characters from words
new_words = fix_words(words);

#1. Find frequency of words and list of prepositions in words
words_freq = Counter(new_words)
prep_idx, prep_count = preposition_count(new_words);

#2. For each preposition, get the N most similar nouns
side = 'most'; #most similar
dm, preposition_word_matches, preposition_word_images, preposition_word_image_pointers, preposition_word_alt_image_pointers, preposition_word_parts, preposition_list, preposition_ids= \
	find_nearest_sentences(caption_data,prep_count,prep_idx,data,coco_all_images,new_words,coco_all_captions,side);

#3. Save data
#Save data as jsons
to_save = [preposition_word_matches,preposition_word_images,\
preposition_word_image_pointers,preposition_word_alt_image_pointers,preposition_word_parts,\
preposition_list,preposition_ids]
names = ['preposition_word_matches','preposition_word_images',\
'preposition_word_image_pointers','preposition_word_alt_image_pointers','preposition_word_parts','preposition_list','preposition_ids']
prepare_json_files(to_save,names)

#Also save dm as a numpy just in case
np.save('dm.npy',dm)
#np.load('dm.npy')

#4. And the N least similar nouns
side = 'least'; #most similar
_, preposition_word_matches, preposition_word_images, preposition_word_image_pointers, preposition_word_alt_image_pointers, preposition_word_parts, preposition_list, preposition_ids= \
	find_nearest_sentences(caption_data,prep_count,prep_idx,data,coco_all_images,new_words,coco_all_captions,side);

#5. Save data
#Save data as jsons
to_save = [preposition_word_matches,preposition_word_images,\
preposition_word_image_pointers,preposition_word_alt_image_pointers,preposition_word_parts,preposition_ids]
names = ['least_preposition_word_matches','least_preposition_word_images',\
'least_preposition_word_image_pointers','least_preposition_word_alt_image_pointers','least_preposition_word_parts','least_preposition_ids']
prepare_json_files(to_save,names)








#2. Sort frequency
#3. Keep only the top N (20?)
#4. Visualize those


#....

#For each preposition, get the N images that most closely match.
	#Identify the nearest Nouns (via parsing). Find images with captions containing the noun and the preposition.
#Then the experiment turns into:
#Each preposition represents a visual reasoning task.
#There are certain images associated with each.
#For each preposition, recover a random sample of 10 associated images + 10 dissassociated images
#Load these into a rapid task, 2 seconds each image, where participants are either cued or not cued to the preposition beforehand, then have to complete the task.
#Analysis of the data could reveal structure of the tasks.
	#Use results + the word2vec distance in a clustering procedure. 





# #Visualize everything for the hell of it...
# tsne = TSNE(n_components=2, init='pca', random_state=0);
# Y = tsne.fit_transform(data);
# ax = fig.add_subplot(121)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("All data")
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
# plt.show()

# #Add cutoff for words -- remove infrequent words
# words_freq = Counter(words);
# cutoff = np.mean(words_freq)# - (2 * np.std(words_freq))
# mask = words_freq > cutoff
# trim_data = data[mask,:]
# Y = tsne.fit_transform(trim_data);
# ax = fig.add_subplot(122)
# plt.title("Frequency trimmed data")
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
# plt.show()


#Given that there's a finite amount of prepositions, first order them by frequency, then view their closest relationships. 
#This will give an idea of the kinds of nouns and verbs typically appearing with each.
#Use these "graphs" as ingredients to experiment generation

#Restrict to the above-mean frequent prepositions.
