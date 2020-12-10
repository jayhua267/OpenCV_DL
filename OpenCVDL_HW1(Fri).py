import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QComboBox
from PIL import Image, ImageOps
from scipy import signal
from scipy import misc




class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Homework1.ui', self)                    
        # pass the definition/method, not the return value!        
        #HW1
        self.LoadImage.clicked.connect(lambda: self.ImageProcessing(1))
        self.Separation.clicked.connect(lambda: self.ImageProcessing(2)) 
        self.Flipping.clicked.connect(lambda: self.ImageProcessing(3)) 
        self.Blending.clicked.connect(lambda: self.ImageProcessing(4)) 

        #HW2
        self.Median.clicked.connect(lambda: self.ImageSmoothing(1))
        self.Gaussian.clicked.connect(lambda: self.ImageSmoothing(2)) 
        self.Bilateral.clicked.connect(lambda: self.ImageSmoothing(3))         

        #HW3
        self.GaussianBlur.clicked.connect(lambda: self.EdgeDetection(1))
        self.SobelX.clicked.connect(lambda: self.EdgeDetection(2)) 
        self.SobelY.clicked.connect(lambda: self.EdgeDetection(3)) 
        self.Magnitude.clicked.connect(lambda: self.EdgeDetection(4)) 

        #HW4
        self.Transformation.clicked.connect(self.findTransformation)

        self.show()

#######-------------HOMEWORK 1--------------------##########

    def nothing(self,x):
        pass

    def ImageProcessing(self,i):
        if i == 1 :
            print("Load Image File Button Pressed : \n")
            im = cv2.imread('Q1_Image/Uncle_Roger.jpg')
            cv2.imshow('image', im)
            h, w, _ = im.shape
            print('width: ', w)
            print('height:', h)
        elif i == 2:
            print("Color Separation Button Pressed : \n")
            im = cv2.imread('Q1_Image/Flower.jpg')
            blue, green, red = cv2.split(im)
            zeros = np.zeros(blue.shape, np.uint8)
            
            blueBGR = cv2.merge((blue,zeros,zeros))
            greenBGR = cv2.merge((zeros,green,zeros))
            redBGR = cv2.merge((zeros,zeros,red))
            
            
            cv2.imshow('blue BGR', blueBGR)
            cv2.imshow('green BGR', greenBGR)
            cv2.imshow('red BGR', redBGR)
        elif i == 3:
            print("Image Flipping Button Pressed : \n")
            im = cv2.imread('Q1_Image/Uncle_Roger.jpg')
            flipHorizontal = cv2.flip(im, 1)
            cv2.imshow('Original Image', im)
            cv2.imshow('Result', flipHorizontal)
        
        else :
            print("Image Bleding Button Pressed : \n")
            im = cv2.imread('Q1_Image/Uncle_Roger.jpg')
            flipHorizontal = cv2.flip(im, 1)
            output = cv2.addWeighted(im, 0.5, flipHorizontal, 0.5, 0)
            cv2.namedWindow('blendWindow')
            cv2.createTrackbar('Blend','blendWindow',0,255,self.nothing)

            while(1):
                cv2.imshow('blendWindow',output)
                k = cv2.waitKey(1) & 0xFF
                #print(k)
                if k != 255:
                    break
                # get current positions of four trackbars
                pos = cv2.getTrackbarPos('Blend','blendWindow')
                output= cv2.addWeighted(im, pos/255, flipHorizontal, 1-pos/255, 0.0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

#######-------------HOMEWORK 2--------------------##########
    def ImageSmoothing(self,i):
        im = cv2.imread('Q2_Image/Cat.png')
        if i == 1 :
            print("Median Filter Button Pressed : \n")
            median = cv2.medianBlur(im,5)
            cv2.imshow('Median Filter', median)            

        elif i == 2 :
            print("Gaussian Blur Button Pressed : \n")
            blur = cv2.GaussianBlur(im,(5,5),0)
            cv2.imshow('Gaussian Blur', blur)

        else:
            print("Bilateral Filter Button Pressed : \n")
            blur = cv2.bilateralFilter(im,9,75,75)
            cv2.imshow('Bilateral Filter', blur)

        cv2.imshow('Original', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#######-------------HOMEWORK 3--------------------##########
    def EdgeDetection(self,i):
        im = cv2.imread('Q3_Image/Chihiro.jpg')
        print("Gaussian Blur Button Pressed : \n")            
        GrayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray Image', GrayIm)

        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        h, w = GrayIm.shape[:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        SmoothIm  = GrayIm.copy()

        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()   
        SoX_Img = SmoothIm.copy()
        SoX_Arr = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   

        SoY_Img = SmoothIm.copy()
        SoY_Arr = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])     
        
        # grad = signal.convolve2d(GrayIm, gaussian_kernel, boundary='symm', mode='same')
        # grad1 = grad.astype('uint8')

        for x in range(1, h-1):
            for y in range(1, w-1):      
                px1 = GrayIm[x-1][y-1]  * gaussian_kernel[0][0]
                px2 = GrayIm[x-1][y]    * gaussian_kernel[0][1]
                px3 = GrayIm[x-1][y+1]  * gaussian_kernel[0][2]
                px4 = GrayIm[x][y-1]    * gaussian_kernel[1][0]
                px5 = GrayIm[x][y]      * gaussian_kernel[1][1]
                px6 = GrayIm[x][y+1]    * gaussian_kernel[1][2]
                px7 = GrayIm[x+1][y-1]  * gaussian_kernel[2][0]
                px8 = GrayIm[x+1][y]    * gaussian_kernel[2][1]
                px9 = GrayIm[x+1][y+1]  * gaussian_kernel[2][2]

                average = (px1 + px2 + px3 + px4 + px5 + px6 + px7 + px8 + px9)
                SmoothIm[x][y] = average

        #Sobel X       
        SoX_Img = signal.convolve2d(SmoothIm, SoX_Arr, boundary='symm', mode='same')
        SoX_Img1 = cv2.convertScaleAbs(SoX_Img) #  0-255
        #Sobel Y
        SoY_Img = signal.convolve2d(SmoothIm, SoY_Arr, boundary='symm', mode='same')
        SoY_Img1 = cv2.convertScaleAbs(SoY_Img)

        #Magnitude    
        Mag_Img = np.sqrt(SoX_Img**2 + SoY_Img**2)
        Mag_Img1 = cv2.convertScaleAbs(Mag_Img)     

        if i == 1:
            cv2.imshow('Smooth Gray Image', SmoothIm)
        if i == 2:
            cv2.imshow('Sobel X Image', SoX_Img1)
        if i == 3:
            cv2.imshow('Sobel Y Image', SoY_Img1)          
        if i == 4:
            cv2.imshow('Magnitude Image', Mag_Img1)    
            # print(np.max(Mag_Img))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

#######-------------HOMEWORK 4--------------------##########
    def findTransformation(self):
        im = cv2.imread('Q4_Image/Parrot.png')
        print("Transformation Button Pressed : \n") 
        # Tx_Old = 160
        # Ty_Old = 84

        RotNew = int(self.RotationBox.text())
        ScaNew = float(self.ScalingBox.text())
        Tx_New = int(self.TxBox.text())
        Ty_New = int(self.TyBox.text())
        print(Tx_New,Ty_New)
        
        rows,cols,ch = im.shape
        # transfer to x y position
        M = np.float32([[1,0,Tx_New],[0,1,Ty_New]])
        dst = cv2.warpAffine(im,M,(cols,rows))
        # rotation image
        M1 = cv2.getRotationMatrix2D((Tx_New+160,Ty_New+84),RotNew,ScaNew)
        dst1 = cv2.warpAffine(dst,M1,(cols,rows))

        plt.subplot(121),plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),plt.title('Input')
        plt.subplot(122),plt.imshow(cv2.cvtColor(dst1,cv2.COLOR_BGR2RGB)),plt.title('Output')
        plt.show()
        
app = QtWidgets.QApplication(sys.argv) 
window = Ui()
app.exec_()

 