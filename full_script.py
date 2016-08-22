#!/usr/bin/env python2.7
import numpy as np
from functions.functions import *

#Model preparation settings
home_dir = '/Users/drewlinsley/Documents/word_games'
data_dir = home_dir + '/data_files'
temp_data_files = home_dir + '/temp_data_files'
nearest_neighbor_image_output_dir = home_dir + '/good_mosaics'
furthest_neighbor_image_output_dir = home_dir + '/bad_mosaics'
word2vec_file = data_dir + '/full_wiki_wangvec.txt'
captions_file = data_dir + '/captions_train2014.npy'
captions_id_file = data_dir + '/captions_train2014ids.npy'
prepared_model_output = data_dir + '/sentences.npy'
prepared_model_captions = data_dir + '/my_captions.npy'

#List of COCO files
coco_array = [data_dir + '/instances_train2014.json', data_dir + '/instances_val2014.json', u'images']

#Words for probing model
load_data = False
word_directory = home_dir + '/prob_word_lists'
preposition_file = word_directory + '/prepositions_verbs.csv'
noun_file = word_directory + '/nouns.csv'

#Settings for printing images
num_ims_to_plot = 25
subplot_arrangement = [5,5]
num_ims_to_save = 100

#Make folders
if not os.path.exists(temp_data_files):
    os.makedirs(temp_data_files)
if not os.path.exists(nearest_neighbor_image_output_dir):
    os.makedirs(nearest_neighbor_image_output_dir)
if not os.path.exists(furthest_neighbor_image_output_dir):
    os.makedirs(furthest_neighbor_image_output_dir)

#Model preparation
caption_ids = np.load(captions_id_file)
d = read_file(word2vec_file,' ') #Load word2vec data
words, data = split_data(d);
if load_data:
	caption_data = np.load(prepared_model_output)
	my_captions = np.load(prepared_model_captions)
else:
	caption_data,my_captions = prepare_sentences(data,words,captions_file,prepared_model_output,prepared_model_captions)

#Get word lists and coco images
prepositions = read_file(preposition_file,',') #Load prepositions
nouns = read_file(noun_file,',') #Load prepositions
coco_all_images = prepare_jsons(coco_array[0],coco_array[1],coco_array[2])
new_words = fix_words(words); #prepare words for word_script_v2

#Find most similar coco captions to each word in prepositions (change this to prepositions + nouns)
if load_data:
	pwi_nearest = read_json(data_dir + '/preposition_word_images.json')
	annotations_nearest = read_json(data_dir + '/preposition_ids.json')
	pwi_furthest = read_json(data_dir + '/least_preposition_word_images.json')
	annotations_furthest = read_json(data_dir + '/least_preposition_ids.json')
else:
	pwi_nearest,_,annotations_nearest,pwi_furthest,_,annotations_furthest = run_coco_finder(new_words,prepositions,nouns,caption_data,caption_ids,data,coco_all_images,my_captions,data_dir)

#Nearest neighbor image mosaics
produce_mosaics(prepositions,pwi_nearest,annotations_nearest,num_ims_to_save,num_ims_to_plot,subplot_arrangement,nearest_neighbor_image_output_dir)

#Furthest neighbor images
produce_mosaics(prepositions,pwi_furthest,annotations_furthest,num_ims_to_save,num_ims_to_plot,subplot_arrangement,furthest_neighbor_image_output_dir)
