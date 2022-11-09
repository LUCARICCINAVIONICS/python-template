#!./NVNN_env/Scripts/python.exe
import platform
from re import S
import preprocessing as prp
import pathlib
import os
import glob
import sys
import subprocess
from shutil import get_terminal_size
import Neural

DATASETPATH = pathlib.Path(pathlib.Path.cwd(),"Dataset")
ORIGINALPATH = pathlib.Path(DATASETPATH,'Origin')
DAYPATH = pathlib.Path(DATASETPATH,'Targets')
NIGHTPATH = pathlib.Path(DATASETPATH,'Labels')

WIDTH = 300
HEIGHT = 300
SIZE = (WIDTH,HEIGHT)
BR = 0.1
COL = 5
SP = 0.1

BUFFERSIZE = 60000
BATCHZISE = 256

class TerminalApp():
    
    preprocessor = prp.preprocessing()
    network = Neural.Network()
    
    def __init__(self):
        mDatasetpath = None
        mOriginalPath = None
        mTargetsPath = None 
        mLablesPath = None
        mFileList = []
        
        self.mHeight = WIDTH
        self.mWidth = HEIGHT
        self.mBr = BR
        self.mSp = COL
        
        self.mBufferSize = BUFFERSIZE
        self.mBatchSize = BATCHZISE
        
        self.CheckFilesPath()
        self.InitPreprocessorParam()
        self.InitNetworkParam()
        
    def InitNetworkParam(self):
        self.network.InitParam(self.mBufferSize,
                               self.mBatchSize,
                               self.mHeight,
                               self.mWidth)
        
    def GetFilePath(self,dataset,original,Targets,Labels):
        self.mDatasetpath = dataset
        self.mOriginalPath = original
        self.mTargetsPath = Targets
        self.mLablesPath = Labels
        
    def CheckFilesPath(self):
        FileList = []
        for p in pathlib.Path( ORIGINALPATH ).iterdir():
            FileList.append( p )
        self.FileList = FileList
            
    def Start(self):
        self.HomeScreen()

    def PrintTitle(self):
        print('------------------------------------------------------------------------------------------')
        print('\n')
        print('                                         NVNN                                             ')
        print('\n')
        print('------------------------------------------------------------------------------------------')

    def InitLog(self):
        self.PrintTitle()
        print('Machine type:          ', platform.machine())
        print('Platform processor:    ', platform.processor())
        print('Platform architecture: ', platform.architecture())
        print('Platform information:  ', platform.platform())
        print('Operating system:      ', platform.system())
        print('System info:           ', platform.system())
        print('------------------------------------------------------------------------------------------')
        input('Press any key to continue ...')
        self.Clear()
        self.HomeScreen()

    def DatasetLog(self):
        self.PrintTitle()
        print("Searching dataset in:            ",DATASETPATH)
        print("Searching original images in:    ",ORIGINALPATH)
        print("Searching target images in:      ",DAYPATH)
        print("Searching processed images in:   ",NIGHTPATH)
        print('\n')


        if len(self.FileList) == 0:
             print("No files found in ",ORIGINALPATH,". \n Check the dataset is present")
             sys.exit("No dataset found")
        else:
            print(len(self.FileList)," files found in ",ORIGINALPATH)
            print('------------------------------------------------------------------------------------------')
        input('Press any key to continue ...')
        self.Clear()
        self.HomeScreen()
   
    def Clear(self): 
        print("\n\n" * get_terminal_size().lines, end='')

    def HomeScreen(self):
        self.PrintTitle()
        print('\n')
        print('    i    ---> show info                d    ---> set dataset path \n')
        print('    h    ---> help                     p    ---> set preprocessing parameters \n')
        print('    n    ---> set network parameters   l    ---> show network \n')
        print('    t    ---> start training           e    ---> exit \n')
        print('    s    ---> start preprocessing                       \n')
        print('------------------------------------------------------------------------------------------')
        selection = input('Choose an option to continue: ')
        match selection:
            case 'i' : 
                self.Clear()
                self.InitLog()
            case 'd' : 
                self.Clear()
                self.DatasetLog()
            case 'e' : 
                self.Clear()
            case 'h' : 
                self.Clear()
                self.HomeScreen()
            case 's':
                self.PreprocessingScreen()
                self.Clear()
                self.HomeScreen()
            case 'p':
                self.SetPreprocessingParametersScreen()
                self.Clear()
                self.HomeScreen()
            case _:
                print(selection + ' is not a valid command')
                self.HomeScreen()
        self.Clear()

    def PreprocessingScreen(self):
        self.PrintTitle()
        l = input('would you like to generate labels? (yes/no) ')
        t = input('would you like to generate targets? (yes/no) ')
        match l:
            case 'yes':
                self.preprocessor.preprocess_labels(self.FileList,self.mLablesPath)
            case _:
                pass
        match t:
            case 'yes':
                self.preprocessor.preprocess_targets(self.FileList,self.mTargetsPath)
            case _:
                pass
        print('------------------------------------------------------------------------------------------\n')
        input('Press any key to continue ...')

    def SetPreprocessingParametersScreen(self):
        self.PrintTitle()
        print("\n")
        print(f"Height:                     {self.mHeight} \n")
        print(f'Width:                      {self.mWidth}\n')
        print(f"Brightness:                 {self.mBr}\n")               
        print(f'Noise:                      {self.mSp} \n')
        ans = input("Would you like to change parameters? (yes,no) ")
        match ans:
            case 'yes':
                self.mHeight = float(input("insert targhet height of the images: "))
                self.mWidth = float(input("insert targhet width of the images: "))
                mSize = (self.mHeight,self.mWidth)
                self.preprocessor.setSize(mSize)
                
                self.mBr = float(input("insert targhet brightness of the images: "))
                self.preprocessor.setBrightness(self.mBr)
                
                self.mSp = float(input("insert targhet noise of the images: "))
                self.preprocessor.setNoise(self.mSp)
            case _:
                input("Press any key to continue ...")
    
    def InitPreprocessorParam(self):
        mSize = (self.mHeight,self.mWidth)
        self.preprocessor.setSize(mSize)
        self.preprocessor.setBrightness(self.mBr)
        self.preprocessor.setNoise(self.mSp)
        
def main():
    App = TerminalApp()
    App.GetFilePath(DATASETPATH,ORIGINALPATH,DAYPATH,NIGHTPATH)
    App.Start()

if __name__ == "__main__" :
    main()
    

