import numpy as np
import tensorflow as tf
from sklearn import svm
import vgg19
import os
import skimage.io, skimage.transform
from sklearn.cross_validation import StratifiedKFold

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

path = "/home/tim/data/examples/dog_positive/aboard/"
imgs_pos = np.array([load_image(os.path.join(path, f)) for f in os.listdir(path)])
path = "/home/tim/data/examples/dog_negative/aboard/"
imgs_neg = np.array([load_image(os.path.join(path, f)) for f in os.listdir(path)])
batch = np.concatenate((imgs_pos, imgs_neg), 0)
labels = np.concatenate((np.ones((imgs_pos.shape[0], )), np.zeros((imgs_neg.shape[0])))).astype(int)

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", batch.shape)
    feed_dict = {images: batch}
    vgg = vgg19.Vgg19("/home/tim/data/vgg19.npy")
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

