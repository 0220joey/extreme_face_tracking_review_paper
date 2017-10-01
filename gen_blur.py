import cv2
import openface
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import pdb
from PIL import ImageEnhance, Image
import os

data_dir = '/media/drive/ibug/300W_cropped/'
dump_dir = '/media/drive/ibug/300W_cropped/synthetic/blur/'
indoor_dump = dump_dir + '01_Indoor'
outdoor_dump = dump_dir + '02_Outdoor'
indoorDF =  pd.read_csv('cropped_indoor_300w.csv')
outdoorDF = pd.read_csv('cropped_outdoor_300w.csv')
globalDF = pd.read_csv('cropped_global_300w.csv')

if __name__ == "__main__":
    i_images = indoorDF['imgPath']
    o_images = outdoorDF['imgPath']
    g_images = globalDF['imgPath']
    iter_count = 0
    stds = [0,1,2,3]
    for std in stds:
        indoor_path = indoor_dump + '_' + str(std) + '/'
        if not os.path.exists(indoor_path):
            os.makedirs(indoor_path)
        for img in i_images:
            rgbImg = cv2.imread(img)
            blur = cv2.GaussianBlur(rgbImg,(15,15),std)
            save_path = indoor_path + img.split('/')[-1]
            cv2.imwrite(save_path,blur)
            iter_count = iter_count + 1
            print("Indoor Std %d with count %d " %(std,iter_count))
        iter_count = 0
        outdoor_path = outdoor_dump + '_' + str(std) + '/'
        if not os.path.exists(outdoor_path):
            os.makedirs(outdoor_path)
        for img in o_images:
            rgbImg = cv2.imread(img)
            blur = cv2.GaussianBlur(rgbImg,(15,15),std)
            iter_count = iter_count + 1
            save_path = outdoor_path + img.split('/')[-1]
            cv2.imwrite(save_path,blur)
            print("Outdoor Std %d with count %d " %(std,iter_count))
