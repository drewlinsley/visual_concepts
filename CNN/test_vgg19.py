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

category = sys.argv[1] 
concept = sys.argv[2] 

lscore = []
path_pos = os.path.join("../../data/imgs/", category, concept, "positive")
imgs_pos = np.array([load_image(os.path.join(path_pos, f)) for f in os.listdir(path_pos)])
path_neg = os.path.join("../../data/imgs/", category, concept, "negative")
imgs_neg = np.array([load_image(os.path.join(path_neg, f)) for f in os.listdir(path_neg)])
batch = np.concatenate((imgs_pos, imgs_neg), 0)
labels = np.concatenate((np.ones((imgs_pos.shape[0], )), np.zeros((imgs_neg.shape[0])))).astype(int)

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", batch.shape)
    feed_dict = {images: batch}
    vgg = vgg19.Vgg19("../../data/vgg19.npy")
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    fc8 = sess.run(vgg.prob, feed_dict=feed_dict)

skf = StratifiedKFold(labels, 5)
lscore = []
for train, test in skf:
     # train a svm on the output
    clf = svm.SVC()
    clf.fit(fc8[train], labels[train])
    lscore.append(clf.score(fc8[test], labels[test]))

print(lscore)
print(np.mean(lscore))
np.save(os.path.join("../../data/res/", category + "_" + concept + ".npy"), lscore)
