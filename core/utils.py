# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 09:39:52 2018

@author: zhou
"""
import h5py
import numpy as np
#import scipy
import os
import cv2


def init_setting():
    
    if os.path.exists('./dataset/train.h5'):
        print("loading the file 'train.h5' ")
        pass
    else:
        preconditioning_data(dataset='./dataset/Train')
        print("creatd the file 'train.h5' ")

        
def pre_setting(image):
    
    #Hyperparameter
    scale=2
    image_size=33
    stride=21
    
#    image=imread(onefile)#'YCrCb'
    height, width, _=image.shape
        
    input_image=cv2.resize(image,(int(width*scale), int(height*scale)),interpolation=cv2.INTER_LINEAR)
    
    height, width, _=input_image.shape

    input_image = input_image.astype(np.float)  / 255
    
    sub_input_sequence=[]

    nx = ny = 0 
    for x in range(0, height-image_size+1, stride):
        nx += 1; ny = 0
        for y in range(0, width-image_size+1, stride):
            ny += 1
            sub_input = input_image[x:x+image_size, y:y+image_size] # [33 x 33 x 3]

            
            sub_input, _, _ = np.split(sub_input, 3, axis=2)
            
            sub_input_sequence.append(sub_input)

            
    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    
    arrCrCb=input_image[:,:,1:3]
    return arrdata, arrCrCb, nx, ny
    
        
def preconditioning_data(dataset, scale=3):
    
    #Hyperparameter
    image_size=33
    label_size=21
    stride=14
    
    filenames = os.listdir(dataset)
    if len(filenames)==0:
        raise ValueError('Without any files in ./dataset/Train')
        
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) / 2#(33-21)/2
    
    for onefile in filenames:
        onefile_path=os.path.join(dataset, onefile)
        
        image=imread(onefile_path)#'YCrCb'
        height, width, _=image.shape
        image=image[0:scale*(height//scale), 0:scale*(width//scale), :]#label_image
        
        height, width, _=image.shape
        input_image=cv2.resize(image,(int(width/scale), int(height/scale)),interpolation=cv2.INTER_AREA)
        input_image=cv2.resize(input_image, (width, height), interpolation=cv2.INTER_CUBIC)

        image = image.astype(np.float) / 255
        input_image = input_image.astype(np.float)  / 255
                
        for x in range(0, height-image_size+1, stride):
            for y in range(0, width-image_size+1, stride):
                sub_input = input_image[x:x+image_size, y:y+image_size] # [33 x 33 x 3]
                sub_label = image[x+int(padding):x+int(padding)+label_size, y+int(padding):y+int(padding)+label_size] # [21 x 21 x 3]

                # Make channel value
                sub_input, _, _ = np.split(sub_input, 3, axis=2)
                sub_label, _, _ = np.split(sub_label, 3, axis=2)

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]
    
    savePath='./dataset/train.h5'
    save2h5File(arrdata, arrlabel, savePath)
        

def merge(imagesY, CrCb, size):
    
    #Hyperparameter
    #input 33 x 33 output 21 x 21
    input_size=33

    h, w = imagesY.shape[1], imagesY.shape[2]
    assert h==w,'imagesY.shape[1]!=imagesY.shape[2]'
    padding=(input_size-h)//2
    
    hc, wc = CrCb.shape[0], CrCb.shape[1]
    CrCb_cutsize_h=int((hc-input_size)/h)*h+input_size
    CrCb_cutsize_w=int((wc-input_size)/w)*w+input_size

    img_CrCb=CrCb[padding:CrCb_cutsize_h+padding, padding:CrCb_cutsize_w+padding, :]
    img_CrCb=img_CrCb*255
    
    
    img = np.zeros((h*size[0], w*size[1], 1))
    for idx, image in enumerate(imagesY):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
        
    img[img > 1.0] = 1.0
    img[img < 0.0] = 0.0
    
    img=img*255

    img_CrCb=cv2.resize(img_CrCb,(w*size[1], h*size[0])).astype(np.float)

    result=np.zeros((h*size[0], w*size[1], 3))

    result[:,:,0]=img[:,:,0]
    result[:,:,1:3]=img_CrCb[:,:,0:2]
    
    return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_YCR_CB2BGR)

    

def read_data(data_path): 

    with h5py.File(data_path, 'r') as file:
        data = np.array(file.get('data'))
        label = np.array(file.get('label'))
        return data, label 

def imread(path):

    #use to opencv lib
    img=cv2.imread(path)#BGR
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

#    return scipy.misc.imread(path, mode='YCbCr').astype(np.float) 

   
def imsave(image, result_path):
    cv2.imwrite(result_path, image)


def save2h5File(data, label, savePath):
        
    with h5py.File(savePath,'w') as file:
        file.create_dataset('data', data=data)
        file.create_dataset('label', data=label)
#        print('data saved')
        
