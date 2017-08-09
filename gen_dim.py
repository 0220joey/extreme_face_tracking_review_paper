import cv2
import openface
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import pdb
from PIL import ImageEnhance, Image
import os

data_dir = '/media/drive/ibug/300W/'
dump_dir = '/media/drive/ibug/300W/synthetic/dim/'
indoor_dump = dump_dir + '01_Indoor'
outdoor_dump = dump_dir + '02_Outdoor'
indoorDF =  pd.read_csv('indoor_300w.csv')
outdoorDF = pd.read_csv('outdoor_300w.csv')
globalDF = pd.read_csv('global_300w.csv')

if __name__ == "__main__":
    i_images = indoorDF['imgPath']
    o_images = outdoorDF['imgPath']
    g_images = globalDF['imgPath']
    iter_count = 0
    factors = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    for factor in factors:
        indoor_path = indoor_dump + '_' + str(factor) + '/'
        if not os.path.exists(indoor_path):
            os.makedirs(indoor_path)
        for img in i_images:
            rgbImg = cv2.imread(img)
            rgbImg = cv2.cvtColor(rgbImg,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(rgbImg)
            enhancer = ImageEnhance.Brightness(pil_im)
            adjusted_img = enhancer.enhance(factor)
            save_path = indoor_path + img.split('/')[-1]
            adjusted_img.save(save_path)
            iter_count = iter_count + 1
            print("Factor %d wiht count %d " %(factor,iter_count))
        iter_count = 0
        outdoor_path = outdoor_dump + '_' + str(factor) + '/'
        if not os.path.exists(outdoor_path):
            os.makedirs(outdoor_path)
        for img in o_images:
            rgbImg = cv2.imread(img)
            iter_count = iter_count + 1
            rgbImg = cv2.cvtColor(rgbImg,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(rgbImg)
            enhancer = ImageEnhance.Brightness(pil_im)
            adjusted_img = enhancer.enhance(factor)
            save_path = outdoor_path + img.split('/')[-1]
            adjusted_img.save(save_path)
            print("Factor %d wiht count %d " %(factor,iter_count))
