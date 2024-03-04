import numpy as np
import glob as glob
import cv2
import os, shutil
from shutil import copy
from PIL import Image

### This records the main operations to preprocess the data in Deepglobe Dataset

################################################################
# step 1: cut the big image and mask into smaller one
################################################################
'''
PATH = '../dataset/Deepglobe/'
PATH_TO_DATA = os.path.join(PATH, '01_train_ori/')
PATH_TO_IMAGE = os.path.join(PATH, '02_train_crop/')
PATH_TO_LABEL = os.path.join(PATH, '02_train_crop/')

for f in os.listdir(PATH_TO_DATA):
    path = PATH_TO_DATA + f.strip()
    img = cv2.imread(path)
    hei = img.shape[0]
    wid = img.shape[1]

    # get 6 roi
    num = 6
    for i in range(0, num):
        for j in range(0, num):
            print(i)
            hei_0 = int(i * hei / num)
            hei_1 = int((i + 1) * hei / num)
            wid_0 = int(j * wid / num)
            wid_1 = int((j + 1) * wid / num)
            roiImg = img[hei_0:hei_1, wid_0:wid_1]
            if f.endswith('.jpg'):
                path = PATH_TO_IMAGE + f.strip()[0:-4] + "_" + str(i) + str(j) + ".jpg"
                cv2.imwrite(path, roiImg)
                print(path)
            else:
                path = PATH_TO_LABEL + f.strip()[0:-4] + "_" + str(i) + str(j) + ".png"
                cv2.imwrite(path, roiImg)
                print(path)



############################################################################
# step 2: filter single color mask and foreground percent small than 0.048
############################################################################

PATH = '../dataset/Deepglobe/'
PATH_TO_DATA = os.path.join(PATH, '02_train_crop/')
PATH_TO_FILTER_DATA = os.path.join(PATH, '03_train_filter/')

path_folder = os.path.expanduser(PATH_TO_DATA)

for f in os.listdir(path_folder):

    if f.endswith('.png'):
        path = PATH_TO_DATA + f.strip()
        img = Image.open(path)
        colors = img.getcolors()
        percent_min = 1
        if len(colors) > 1:
            for num in colors:
                percent = num[0] / 166464
                if percent < percent_min:
                    percent_min = percent
            if percent_min > 0.048:
                path = PATH_TO_FILTER_DATA + f.strip()
                img.save(path)
                print(path)

############################################################################
# step 3: generate binary mask for each class
############################################################################

labelset = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255]]

def GetBinaryMap(img, label):
    mask = np.ones((img.shape))
    isnull = 1
    for j in range(len(img)):
        for k in range(len(img)):
            if (img[j][k].tolist() == labelset[label]):
                mask[j][k] = [255, 255, 255]
                isnull = 0
            else:
                mask[j][k] = [0, 0, 0]
    return isnull, mask

# list all files under the folder

PATH = '../dataset/Deepglobe/'
oridir = os.path.join(PATH, '03_train_filter/')
masklist = os.listdir(oridir)
desdir = os.path.join(PATH, '04_train_cat/')

for label in range(0, 6):
    filename = []# to save the filename which belongs to this class
    desdir_label = os.path.join(desdir, str(label+1))
    for i in range(len(masklist)):
        if (masklist[i].endswith('.png')):
            imgfile = os.path.join(oridir, masklist[i])
            img = cv2.imread(imgfile, 1)
            isnull, binary_mask = GetBinaryMap(img, label) ## isnull: 1 denotes whole mask is black
            if (isnull == 0):
                filename.append(os.path.splitext(masklist[i])[0])
                cv2.imwrite(desdir_label + '/test/groundtruth/' + masklist[i], binary_mask)
                print(masklist[i])
                print(label)
    file = open(desdir + str(label+1) + '.txt', 'w')
    for n in range(len(filename)):
        file.write(str(filename[n]))
        file.write('\n')
    file.close()

'''

############################################################################
# step 4: copy corresponding images of each mask
############################################################################

PATH = '../dataset/Deepglobe/'
PATH_TO_IMG = os.path.join(PATH, '02_train_crop/')
PATH_TO_MSK = os.path.join(PATH, '04_train_cat/')

for cat in range(6):
    cat_path = os.path.join(PATH_TO_MSK, str(cat+1), 'test', 'groundtruth')
    des_path = os.path.join(PATH_TO_MSK, str(cat+1), 'test', 'origin')
    for msk in os.listdir(cat_path):
        img = msk[:-12] + '_sat_' + msk[-6:-4] + '.jpg'
        img_path = os.path.join(PATH_TO_IMG, img)
        copy(img_path, des_path)
        print(img_path, des_path)
        print(cat)