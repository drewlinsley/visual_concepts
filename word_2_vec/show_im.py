import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
import numpy as np
sys.path.append('../../coco/PythonAPI')
from pycocotools.coco import COCO
import skimage.io as io

path = "../../data/"
# load caption and images ids
coco_caps = COCO(os.path.join(path, 'coco', 'annotations', 'captions_train2014.json'))
annIds = coco_caps.getAnnIds()

# load concept
bc = np.loadtxt(os.path.join(path, 'best_concepts.txt'), type('str'))

# load image and display + annotation + concept
for nim in range(10):
    print(nim)
    ann = coco_caps.loadAnns(annIds)[nim]
    img = coco_caps.loadImgs(ann['image_id'])[0]
    I = io.imread(os.path.join(path, 'coco', 'train2014', img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    plt.savefig(os.path.join(path, 'imgs', 'imgs_'+str(nim)+'.png'))
    sts = coco_caps.showAnns([ann])
    print(bc[nim])


