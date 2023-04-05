# -*- coding: utf-8 -*-
import cv2
import numpy as np
import harris
import PIL.Image
import PIL.ExifTags
from mfcc_talkbox import mfcc	

def harris_features(im):
    response = cv2.cornerHarris(im, 7, 5, 0.05)
    points = harris.get_harris_points(response)
    desc = harris.get_descriptors(im, points)
    return points, desc

def get_harris_features(im_list):
    total = len(im_list)
    print('Generating Harris features for [', total, '] images ...')
    features = {}
    count = 0
    for im_name in im_list:
        im = cv2.imread(im_name, 0)
        points, desc = harris_features(im)
        features[im_name] = np.array(desc)
        count += 1
    return features

def colorhist(im):
    chans = cv2.split(im)
    color_hist = np.zeros((256,len(chans)))
    for i in range(len(chans)):
        color_hist[:,i] = np.histogram(chans[i], bins=np.arange(256+1))[0]/float((chans[i].shape[0]*chans[i].shape[1]))
    return color_hist


def get_colorhist(im_list):
    total = len(im_list)
    print('Generating ColorHist features for [', total, '] images ...')
    features = {}
    count = 0
    for im_name in im_list:
        im = cv2.imread(im_name)
        color_hist = colorhist(im)
        features[im_name] = color_hist
        count += 1
    return features

def get_sift_features(im_list):
    """get_sift_features accepts a list of image names and computes the sift descriptos for each image. It returns a dictionary with descriptor as value and image name as key """
    sift = cv2.xfeatures2d.SIFT_create()
    features = {}
    total = len(im_list)
    count = 0
    print('Generating SIFT features for [', total, '] images ...')
    for im_name in im_list:
        # load grayscale image
        im = cv2.imread(im_name, 0)
        kp, desc = sift.detectAndCompute(im, None)
        features[im_name] = desc
        count += 1
    return features
    
# extract tags
def extract_tags(filename):
    try:
        print('tags for', filename)
        img = PIL.Image.open(filename)
        exif_data = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in list(img._getexif().items())
            if k in PIL.ExifTags.TAGS
        }
        print(exif_data['UserComment'])
        tags = exif_data['UserComment'].split(',')
        tags = [t.strip() for t in tags]
        return tags
    except:
        print('No tags could be found for: ' + filename)
        return []
    
    
def extract_mfcc(audio_samples, fs):
    # find the smallest non-zero sample in both channels
    #nonzero = min(min([abs(x) for x in audio_samples[:,0] if abs(x) > 0]), min([abs(x) for x in audio_samples[:,1] if abs(x) > 0]))
    audio_samples = audio_samples.copy()
    window_time_length=0.01
    window_samples_length=int(fs*window_time_length)
    nonzero = 1
    audio_samples[audio_samples==0] = nonzero
    return mfcc(audio_samples, fs=fs,nwin=window_samples_length, nfft=window_samples_length*2)

