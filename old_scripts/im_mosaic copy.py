#Run wang2vec
#Then prepare_sentences
#Followed by word_script_v2
#And finally im_mosaic

#Create image mosaics 
import readmagick
import pylab
import json 
import os
import urllib
import glob
import skimage.io as io
import matplotlib.pyplot as plt
import csv

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
def produce_mosaics(prep_list,pwi_file,annotations,output_dir,temp_im_dir):
	num_preps = len(prep_list)
	dl_file = urllib.URLopener()
	for idx in range(0,num_preps):
		cur_url_list = pwi_file[idx]
		cur_annotations = annotations[idx]
		#backup_cur_url_list = back_pwi_file[idx]
		num_urls = len(cur_url_list)
		i = 0
		il = 1
		while il < len(cur_url_list) and il < num_ims_to_plot:
			try:
				dl_name = temp_im_dir + '/' + str(il - 1) + '.jpg'
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
		pylab.savefig(output_dir + '/' + prep_list[idx] + '.jpg', format='jpg', dpi=1000)
		remove_files(temp_im_dir + '/*')

num_preps = 10
num_ims_to_plot = 25
subplot_arrangement = [5,5]
home_dir = '/Users/drewlinsley/Documents/word_games'
temp_im_dir = home_dir + '/temp_im_dir'
output_dir = home_dir + '/good_mosaics'

#Read preposition list
#prep_list = read_json('preposition_list.json')
#prep_list = read_json('preposition_list.json')
pl = read_file('prepositions_verbs.csv',',') #Load prepositions
prep_list = []
for i in pl:
	prep_list.append(i[0])

#Make folders
output_dir = home_dir + '/good_mosaics'
if not os.path.exists(temp_im_dir):
    os.makedirs(temp_im_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Read image names -- closest
#pwi_file = read_json('preposition_word_alt_image_pointers.json')
pwi_file = read_json('preposition_word_images.json')
#Read annotation list
annotations = read_json('preposition_ids.json')
produce_mosaics(prep_list,pwi_file,annotations,output_dir,temp_im_dir)

#Read image names -- furthest
output_dir = home_dir + '/bad_mosaics'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#pwi_file = read_json('least_preposition_word_alt_image_pointers.json')
pwi_file = read_json('least_preposition_word_images.json')
#Read annotation list
annotations = read_json('least_preposition_ids.json')
produce_mosaics(prep_list,pwi_file,annotations,output_dir,temp_im_dir)





