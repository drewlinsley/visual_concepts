#!/usr/bin/env python2.7
from __future__ import division, print_function

import sys
import numpy as np
from collections import Counter
import csv
import scipy.spatial.distance as dis
import re
import operator
import codecs
import json 
import readmagick
import pylab
import json 
import os
import urllib
import glob
import skimage.io as io
import matplotlib.pyplot as plt

def read_file(name, deli):
	with open(name,'r') as f:
		reader=csv.reader(f,delimiter=deli,quoting=csv.QUOTE_NONE)
		out = list(reader)
	return out

def read_json(name):
	with open(name,) as f:
		out=json.load(f)
	return out

def remove_files(dir):
	files = glob.glob(dir)
	for f in files:
	    os.remove(f)

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

def preposition_count(new_words,prepositions):
	prep_idx = []
	prep_count = []
	for idx in prepositions:
		ics = index_containing_substring(new_words,idx[0])
		if ics == -1:
			print(str(idx) + ' is not found in coco')
		prep_idx.append(ics)
		prep_count.append(new_words.count(idx[0]))
		#For some reason some of these appear multiple times... multiple word2vec "contexts" I guess?
	prep_idx = np.asarray(prep_idx)
	prep_count = np.asarray(prep_count)
	#prep_mask = prep_idx != -1;
	#prep_idx = prep_idx[prep_mask];
	#prep_count = prep_count[prep_mask];
	return prep_idx, prep_count

def save_json(input,path):
	json.dump(input, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

def prepare_json_files(to_save,names,output_dir):
	for i in range(0,len(to_save)):
		path = output_dir + '/' + names[i] + ".json"
		save_json(to_save[i],path)

def prepare_sentences(data,words,captionsfile,output_file,caption_output_file):

	#Reads sentences and Word2vec vector space. Accumulates the average vector for words in each sentence. These are used to represent sentences.
	my_captions = np.load(captionsfile)
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

	np.save(output_file,sentences)
	np.save(caption_output_file,my_captions)
	return sentences,my_captions

def old_find_nearest_sentences(caption_data,caption_ids,prep_count,prep_idx,noun_idx,noun_count,data,coco_all_images,new_words,coco_all_captions,side):
	#2. For each preposition, get the N most/least similar nouns
	#a. precalculate distance between each preposition and all other words
	if side == 'high': #high dissimilarity
		side = -1
		print('Using side ' + str(side))
	else:
		side = 1 #low dissimilarity
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
	#m = 200;


	#Find average animal vector
	average_noun = np.zeros((data.shape[1]))
	for idx in range(0,noun_idx.shape[0]):
		average_noun += data[noun_idx[idx],:]
	average_noun/=noun_idx.shape[0]

	#1. Previously stored the image_ids of each sentence annotation [caption_ids]
	#2. Unravel image ids from coco-images -- use these later for a lookup table
	#3. For each nearest neighbor caption, coco_ids.index(caption_ids[nearest_neighbor[i]])
	coco_ids = []
	for idx in coco_all_images:
		coco_ids.append(idx[u'id'])

	#Evaluate argmax[1...N](sentences|prep) -- find the N most likely sentences given each preposition.
	total_props = prep_count.shape[0]
	for idx in range(0,total_props):
		print('******Gathering nearest neighbors for preposition ' + str(idx) + '/' + str(total_props) + ': ' + new_words[prep_idx[idx]])
		prep_list.append(new_words[prep_idx[idx]])
		for il in range(0,caption_data.shape[0]):
			#dm[0,il] = dis.cosine(data[prep_idx[idx],:],caption_data[il])
			dm[0,il] = dis.cosine(np.mean([data[prep_idx[idx],:],average_noun],axis=0),\
				caption_data[il])		###### Control for nouns around here... take mean or sum of noun plus the preposition
		#1. For each preposition, find the POS of the nearest and furthest (Catch trials) n words 
		nearest_ids = np.argsort(dm[0,::side])[:n]
		nearest_trans_ids = caption_ids[nearest_ids]
		#Preallocate
		it_word_matches = []
		it_images = []
		it_image_pointers = []
		it_alt_image_pointers = []
		it_word_pos = []
		it_ids = []
		for il in range(0,n):
			im_id = coco_ids[coco_ids.index(nearest_trans_ids[il])]
			it_images.append(u'http://mscoco.org/images/' + str(im_id))
			#it_images.append(u'http://mscoco.org/images/' + str(my_captions[nearest_ids[il]]))
			####
			#it_images.append(coco_all_images[im_id][u'file_name'])
			#it_image_pointers.append(coco_all_images[im_id][u'flickr_url'])
			#it_alt_image_pointers.append(coco_all_images[im_id][u'coco_url'])
			it_ids.append(coco_all_captions[nearest_ids[il]])
		preposition_word_images.append(it_images)
		preposition_ids.append(it_ids)
	return dm, preposition_word_images, prep_list, preposition_ids


##Find coco sentences related to user-supplied words
def working_find_nearest_sentences(caption_data,caption_ids,prep_count,prep_idx,noun_idx,noun_count,data,coco_all_images,new_words,coco_all_captions,side):
	#2. For each preposition, get the N most/least similar nouns
	#a. precalculate distance between each preposition and all other words
	if side == 'high': #high dissimilarity
		side = -1
		print('Using side ' + str(side))
	else:
		side = 1 #low dissimilarity
		print('Using side ' + str(side))

	preposition_word_matches = []
	preposition_word_images = []
	preposition_word_image_pointers = []
	preposition_word_alt_image_pointers = []
	preposition_ids = []
	word_pos = []
	prep_list = []
	n = 50;
	m = int(1*1e3);

	#Find average animal vector
	average_noun = np.zeros((data.shape[1]))
	for idx in range(0,noun_idx.shape[0]):
		average_noun += data[noun_idx[idx],:]
	average_noun/=noun_idx.shape[0]

	#1. Previously stored the image_ids of each sentence annotation [caption_ids]
	#2. Unravel image ids from coco-images -- use these later for a lookup table
	#3. For each nearest neighbor caption, coco_ids.index(caption_ids[nearest_neighbor[i]])
	coco_ids = []
	for idx in coco_all_images:
		coco_ids.append(idx[u'id'])

	#Evaluate argmax[1...N](sentences|prep) -- find the N most likely sentences given each preposition.
	total_props = prep_count.shape[0]
	for idx in range(0,total_props):
		print('******Gathering nearest neighbors for preposition ' + str(idx) + '/' + str(total_props) + ': ' + new_words[prep_idx[idx]])
		prep_list.append(new_words[prep_idx[idx]])
		#First sort for visual content (nouns)
		dm = np.empty((caption_data.shape[0]))
		for il in range(0,caption_data.shape[0]):
			dm[il] = dis.cosine(average_noun,caption_data[il])
		nearest_nouns = np.argsort(dm[::-1])[:m]

		#Choose the m best (lowest distance) then sort for the visual routine (verb/preposition)
		dm = np.empty((nearest_nouns.shape[0]))
		for il in range(0,nearest_nouns.shape[0]):
			dm[il] = dis.cosine(data[prep_idx[idx],:],caption_data[nearest_nouns[il]])
			#dm[0,il] = dis.cosine(np.mean([data[prep_idx[idx],:],average_noun],axis=0),\
			#	caption_data[il])		###### Control for nouns around here... take mean or sum of noun plus the preposition
		#1. For each preposition, find the POS of the nearest and furthest (Catch trials) n words 
		nearest_ids = nearest_nouns[np.argsort(dm[::side])[:n]]
		nearest_trans_ids = caption_ids[nearest_ids]
		#Preallocate
		it_word_matches = []
		it_images = []
		it_image_pointers = []
		it_alt_image_pointers = []
		it_word_pos = []
		it_ids = []
		for il in range(0,n):
			im_id = coco_ids[coco_ids.index(nearest_trans_ids[il])] #WTF nearest_trans vs. nearest_ids
			it_images.append(u'http://mscoco.org/images/' + str(im_id))
			#it_images.append(u'http://mscoco.org/images/' + str(my_captions[nearest_ids[il]]))
			####
			#it_images.append(coco_all_images[im_id][u'file_name'])
			#it_image_pointers.append(coco_all_images[im_id][u'flickr_url'])
			#it_alt_image_pointers.append(coco_all_images[im_id][u'coco_url'])
			it_ids.append(coco_all_captions[nearest_ids[il]])
		preposition_word_images.append(it_images)
		preposition_ids.append(it_ids)
	return dm, preposition_word_images, prep_list, preposition_ids

##Find coco sentences related to user-supplied words
def find_nearest_sentences(caption_data,caption_ids,prep_count,prep_idx,noun_idx,noun_count,data,coco_all_images,new_words,coco_all_captions,side):
	#2. For each preposition, get the N most/least similar nouns
	#a. precalculate distance between each preposition and all other words
	if side == 'high': #high dissimilarity
		side = -1
		print('Using side ' + str(side))
	else:
		side = 1 #low dissimilarity
		print('Using side ' + str(side))

	preposition_word_matches = []
	preposition_word_images = []
	preposition_word_image_pointers = []
	preposition_word_alt_image_pointers = []
	preposition_ids = []
	word_pos = []
	prep_list = []
	n = 50;
	m = int(1*1e3);

	#Find average animal vector
	average_noun = np.zeros((data.shape[1]))
	for idx in range(0,noun_idx.shape[0]):
		average_noun += data[noun_idx[idx],:]
	average_noun/=noun_idx.shape[0]

	#1. Previously stored the image_ids of each sentence annotation [caption_ids]
	#2. Unravel image ids from coco-images -- use these later for a lookup table
	#3. For each nearest neighbor caption, coco_ids.index(caption_ids[nearest_neighbor[i]])
	coco_ids = []
	for idx in coco_all_images:
		coco_ids.append(idx[u'id'])

	#Evaluate argmax[1...N](sentences|prep) -- find the N most likely sentences given each preposition.
	total_props = prep_count.shape[0]
	for idx in range(0,total_props):
		print('******Gathering nearest neighbors for preposition ' + str(idx) + '/' + str(total_props) + ': ' + new_words[prep_idx[idx]])
		prep_list.append(new_words[prep_idx[idx]])
		if side == 1:
			dm = np.empty((caption_data.shape[0]))
			for il in range(0,caption_data.shape[0]):
				#dm[0,il] = dis.cosine(data[prep_idx[idx],:],caption_data[il])
				dm[il] = dis.correlation((average_noun + data[prep_idx[idx],:]),caption_data[il])		###### Control for nouns around here... take mean or sum of noun plus the preposition
			#1. For each preposition, find the POS of the nearest and furthest (Catch trials) n words 
			nearest_ids = np.argsort(dm[::1])[:n]
			nearest_trans_ids = caption_ids[nearest_ids]
		elif side == -1:
			dm = np.empty((caption_data.shape[0]))
			for il in range(0,caption_data.shape[0]):
				#dm[0,il] = dis.cosine(data[prep_idx[idx],:],caption_data[il])
				dm[il] = dis.correlation((average_noun - data[prep_idx[idx],:]),caption_data[il])		###### Control for nouns around here... take mean or sum of noun plus the preposition
			#1. For each preposition, find the POS of the nearest and furthest (Catch trials) n words 
			nearest_ids = np.argsort(dm[::1])[:n]
			nearest_trans_ids = caption_ids[nearest_ids]

		#Preallocate
		it_word_matches = []
		it_images = []
		it_image_pointers = []
		it_alt_image_pointers = []
		it_word_pos = []
		it_ids = []
		for il in range(0,n):
			im_id = coco_ids[coco_ids.index(nearest_trans_ids[il])] #WTF nearest_trans vs. nearest_ids
			it_images.append(u'http://mscoco.org/images/' + str(im_id))
			it_ids.append(coco_all_captions[nearest_ids[il]])
		preposition_word_images.append(it_images)
		preposition_ids.append(it_ids)
	return dm, preposition_word_images, prep_list, preposition_ids

def run_coco_finder(new_words,prepositions,nouns,caption_data,caption_ids,data,coco_all_images,my_captions,output_dir):
	#1. Find frequency of words and list of prepositions in words
	words_freq = Counter(new_words)
	prep_idx, prep_count = preposition_count(new_words,prepositions);

	#1a. do this for nouns too
	noun_idx, noun_count = preposition_count(new_words,nouns);

	#2. For each preposition, get the N most similar nouns
	side = 'low'; #most
	_, preposition_word_images, preposition_list, preposition_ids= \
		find_nearest_sentences(caption_data,caption_ids,prep_count,prep_idx,noun_idx, noun_count, data,coco_all_images,new_words,my_captions,side);

	#3. Save data
	to_save = [preposition_word_images,\
	preposition_list,preposition_ids]
	names = ['preposition_word_images',\
	'preposition_list','preposition_ids']
	prepare_json_files(to_save,names,output_dir)

	#4. And the N least similar nouns -- REFACTOR THIS CODE
	side = 'high'; #least similar
	_, least_preposition_word_images, least_preposition_list, least_preposition_ids= \
		find_nearest_sentences(caption_data,caption_ids,prep_count,prep_idx,noun_idx, noun_count, data,coco_all_images,new_words,my_captions,side);

	#5. Save data
	to_save = [least_preposition_word_images,\
	least_preposition_list,least_preposition_ids]
	names = ['least_preposition_word_images',\
	'least_preposition_list','least_preposition_ids']
	prepare_json_files(to_save,names,output_dir)

	return preposition_word_images, preposition_list, preposition_ids, least_preposition_word_images, least_preposition_list, least_preposition_ids

def produce_mosaics(prep_list,pwi_file,annotations,num_ims_to_save,num_ims_to_plot,subplot_arrangement,output_dir):

	num_preps = len(pwi_file) - 1 #FIX FOR NOW ... THE LEN IS 1 off for some reason
	dl_file = urllib.URLopener()
	for idx in range(0,num_preps):
		new_folder = output_dir + '/' + prep_list[idx][0]
		if not os.path.exists(new_folder):
		    os.makedirs(new_folder)
		cur_url_list = pwi_file[idx]
		cur_annotations = annotations[idx]
		#backup_cur_url_list = back_pwi_file[idx]
		num_urls = len(cur_url_list)
		i = 0
		il = 1
		while il < len(cur_url_list) and il < num_ims_to_plot:
			try:
				dl_name = new_folder + '/' + str(il - 1) + '.jpg'
				#try:
				#dl_file.retrieve(cur_url_list[il], dl_name)
				response = urllib.urlopen(cur_url_list[il])
				content = response.read()
				f = open(dl_name,'w')
				f.write(content)
				f.close()
				#except Exception, e:
					#dl_file.retrieve(backup_cur_url_list[il], dl_name)
				im = readmagick.readimg(dl_name)
				ax = pylab.subplot(subplot_arrangement[0],subplot_arrangement[1],il,frameon=False, xticks=[], yticks=[])
				str_list = [x.encode('UTF8') for x in cur_annotations[il]]
				ax.set_title(' '.join(str_list),fontsize=4, y=1.08)
				pylab.imshow(im)
				print("Finished image #" + str(il))
			except Exception, g:
				print("Skipping image #" + str(il))
			il += 1
		pylab.savefig(output_dir + '/' + prep_list[idx][0] + '.jpg', format='jpg', dpi=1000)
		#remove_files(temp_im_dir + '/*')

def produce_images(prep_list,pwi_file,annotations,num_ims_to_plot,subplot_arrangement,output_dir,temp_im_dir):
	num_preps = len(pwi_file) - 1 #FIX FOR NOW ... THE LEN IS 1 off for some reason
	dl_file = urllib.URLopener()
	for idx in range(0,num_preps):
		new_folder = output_dir + '/' + prep_list[idx][0]
		if not os.path.exists(new_folder):
		    os.makedirs(new_folder)
		cur_url_list = pwi_file[idx]
		cur_annotations = annotations[idx]
		#backup_cur_url_list = back_pwi_file[idx]
		num_urls = len(cur_url_list)
		i = 0
		il = 1
		while il < len(cur_url_list) and il < num_ims_to_plot:
			try:
				dl_name = new_folder + '/' + str(il - 1) + '.jpg'
				response = urllib.urlopen(cur_url_list[il])
				content = response.read()
				f = open(dl_name,'w')
				f.write(content)
				f.close()
				print("Finished image #" + str(il))
			except Exception, g:
				print("Skipping image #" + str(il))
			il += 1
			#At some point integrate this into the mosaic code... wasteful to do this twice



