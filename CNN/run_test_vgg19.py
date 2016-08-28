import numpy as np
import tensorflow as tf
from sklearn import svm
import vgg19
import os, sys
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # if black and white, duplicate the channels
    if len(img.shape)==2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

path_dat = "../../data"
prep = np.loadtxt('../prob_word_lists/reduced_prepositions.csv', type('str'))
cats = np.loadtxt(os.path.join(path_dat, 'categories.txt'), type('str'))
limgs = []
llabs = []
for concept in prep:
    print(concept)
    for category in cats: 
        path_pos = os.path.join("../../data/imgs/", category, concept, "positive")
        limgs.extend([load_image(os.path.join(path_pos, f)) for f in os.listdir(path_pos)])
        path_neg = os.path.join("../../data/imgs/", category, concept, "negative")
        limgs.extend([load_image(os.path.join(path_neg, f)) for f in os.listdir(path_neg)])
        llabs.extend(np.concatenate((np.ones((len(os.listdir(path_pos)), )), np.zeros((len(os.listdir(path_neg)), )))).astype(int))
limgs = np.array(limgs)
labels = np.array(llabs)

batch_size = 50
lfc8 = []
with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.95)))) as sess:
    vgg = vgg19.Vgg19("../../data/vgg19.npy")
    images = tf.placeholder("float", (batch_size, limgs.shape[1], limgs.shape[2], limgs.shape[3]))
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    for i in range(limgs.shape[0]/batch_size):
        print("batch number " + str(i))
        batch = limgs[i*batch_size:batch_size*(i+1)]
        feed_dict = {images: batch}
        lfc8.append(sess.run(vgg.prob, feed_dict=feed_dict))

np.save(os.path.join(path_dat, 'output_vgg19.npy'), np.array(lfc8))
np.save(os.path.join(path_dat, 'labels_vgg19.npy'), np.array(labels))

