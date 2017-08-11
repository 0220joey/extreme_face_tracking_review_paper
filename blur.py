import cv2
import openface
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import pdb
import os

data_dir = '/media/drive/ibug/300W/synthetic/blur/'
dump_dir = '/media/drive/ibug/300W/results/blur/'
indoor_dump_dir = dump_dir + '01_Indoor_'
outdoor_dump_dir = dump_dir + '02_Outdoor_'
indoor = data_dir + '01_Indoor_'
outdoor = data_dir + '02_Outdoor_'
indoorDF =  pd.read_csv('indoor_300w.csv')
outdoorDF = pd.read_csv('outdoor_300w.csv')
globalDF = pd.read_csv('global_300w.csv')
dlib_model ='/home/joey/openface/models/dlib/shape_predictor_68_face_landmarks.dat'

def clean_points(pts):
    pts = pts.replace("\n","")
    pts = pts.replace("(","")
    pts = pts.replace(")","")
    pts = pts.replace("[","")
    pts = pts.replace("]","")
    pts = pts.replace("array","")
    pts = pts.replace(" ","")
    clean = np.fromstring(pts,sep=',').reshape(68,2)
    return clean

def get_landmarks(model,rgbimg):
    bb = model.getLargestFaceBoundingBox(rgbimg)
    try:
        landmarks = model.findLandmarks(rgbimg,bb)
    except:
        return None
    return np.asarray(landmarks)

def get_avg_dist(pt_labels,landmarks,norm):
    total_dist = 0
    for pt,pred in zip(pt_labels,landmarks):
        dist = np.linalg.norm(pt - pred)
        total_dist = total_dist + dist
    avg_dist = np.divide(total_dist,68)
    return np.divide(avg_dist,norm)

if __name__ == "__main__":
    i_images = indoorDF['imgPath']
    i_points = indoorDF['points']
    o_images = outdoorDF['imgPath']
    o_points = outdoorDF['points']
    g_images = globalDF['imgPath']
    g_points = globalDF['points']
    face_model = openface.AlignDlib(dlib_model)
    total_indoor = 0
    total_outdoor = 0
    resultsDF = pd.DataFrame(columns=('experiment','type','loss',
    'total_detected'))
    iter_count = 0
    factor_count = 0
    tot_indoor_det = 0
    tot_outdoor_det = 0
    factors = [0,1,2,3]
    for factor in factors:
        indoor_dump = indoor_dump_dir + str(factor) + '/'
        if not os.path.exists(indoor_dump):
            os.makedirs(indoor_dump)
        for img,pts in zip(i_images,i_points):
            img = indoor + str(factor) + '/' + img.split('/')[-1]
            rgbImg = cv2.imread(img)
            landmarks = get_landmarks(face_model,rgbImg)
            if landmarks is None:
                print("Failed to find face on %s" %(img.split('/')[-1]))
                continue
            cleaned = clean_points(pts)
            tot_indoor_det = tot_indoor_det + 1
            for (x, y) in landmarks:
                cv2.circle(rgbImg, (x, y), 1, (0, 0, 255), -1)
            for (x, y) in cleaned:
                cv2.circle(rgbImg, (int(x), int(y)), 1, (0, 255, 0), -1)
            dump_path = indoor_dump + img.split('/')[-1]
            cv2.imwrite(dump_path,rgbImg)
            left_inner = landmarks[39]
            right_inner = landmarks[42]
            inter_occ_dist = np.linalg.norm(left_inner - right_inner)
            img_dist = get_avg_dist(cleaned,landmarks,inter_occ_dist)
            total_indoor = total_indoor + img_dist
            iter_count = iter_count + 1
            print(iter_count)
        iter_count = 0
        resultsDF.loc[factor_count] = ['blur','indoor_'+str(factor),total_indoor,tot_indoor_det]
        factor_count = factor_count + 1
        outdoor_dump = outdoor_dump_dir + str(factor) + '/'
        if not os.path.exists(outdoor_dump):
            os.makedirs(outdoor_dump)
        for img,pts in zip(o_images,o_points):
            img = outdoor + str(factor) + '/' + img.split('/')[-1]
            rgbImg = cv2.imread(img)
            landmarks = get_landmarks(face_model,rgbImg)
            if landmarks is None:
                print("Failed to find face on %s" %(img.split('/')[-1]))
                continue
            cleaned = clean_points(pts)
            for (x, y) in landmarks:
                cv2.circle(rgbImg, (x, y), 1, (0, 0, 255), -1)
            for (x, y) in cleaned:
                cv2.circle(rgbImg, (int(x), int(y)), 1, (0, 255, 0), -1)
            tot_outdoor_det = tot_outdoor_det + 1
            dump_path = outdoor_dump + img.split('/')[-1]
            cv2.imwrite(dump_path,rgbImg)
            left_inner = landmarks[39]
            right_inner = landmarks[42]
            inter_occ_dist = np.linalg.norm(left_inner - right_inner)
            img_dist = get_avg_dist(cleaned,landmarks,inter_occ_dist)
            total_outdoor = total_outdoor + img_dist
            iter_count = iter_count + 1
            print(iter_count)
        resultsDF.loc[factor_count] = ['blur','outdoor_'+str(factor),total_outdoor, tot_outdoor_det]
        factor_count = factor_count + 1
    resultsDF.to_csv('blur_results.csv',mode='a',header=True)
