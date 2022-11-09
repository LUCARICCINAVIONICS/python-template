#!./NVNN_env/Scripts/python.exe
import shutil 
import pathlib 
import cv2 
import numpy as np
from PIL import Image ,ImageEnhance
import random


WIDTH = 300
HEIGHT = 300
SIZE = (WIDTH,HEIGHT)
BR = 0.1
COL = 5
SP = 0.1

class preprocessing():
    
    def __init__(self):
        mSize = SIZE
        mCol = COL
        mSp = SP 
        mBr = 0
    
    def preprocess_targets(self,FileList,targets):
        i = 0
        for file in FileList :
            targetname = str(i)+'.jpg'
            target = pathlib.Path(targets,targetname)
            img =  Image.open(file.as_posix())
            img = self.preprocess_target(img,self.mSize)
            print(file,'    ---->    ',target)
            img.save(target)
            i = i + 1

    def preprocess_labels(self,FileList,labels):
        i = 0
        for file in FileList :
            labelname = str(i)+'.jpg'
            label = pathlib.Path(labels,labelname)
            img =  Image.open(file.as_posix())
            img = self.preprocess_label(img,self.mSize,self.mBr,self.mCol,self.mSp)
            print(file,'    ---->    ',label)
            img.save(label)
            i = i + 1
                
                
    def preprocess_label(self,img,size,bright,col,sp):
        img = img.convert('L')
        img = img.resize(size)
        Brightness = ImageEnhance.Brightness(img)
        img = Brightness.enhance(bright)
        Color = ImageEnhance.Color(img)
        img = Color.enhance(col)
        noise = self.sp_noise(sp,size)
        addition = noise + np.asarray(img)
        return Image.fromarray(addition)

    def sp_noise(self,sp,size):
        noise= np.zeros(size,dtype=np.uint8)
        noise = cv2.randu(noise,0,sp*255)
        return noise

    def preprocess_target(self,img,size):
        img = img.convert('L')
        img = img.resize(size)
        return img

    def setSize(self,size):
        self.mSize = size
        
    def setColorIntensity(self,col):
        self.mCol = col
    
    def setBrightness(self,br):
        self.mCr = br
        
    def setNoise(self,sp):
        self.mSp = sp