import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QComboBox


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Homework1.ui', self)                    
        # pass the definition/method, not the return value!
        self.Corners.clicked.connect(self.findCornerPress) 
        self.Intrinsic.clicked.connect(self.findIntrinsicPress)
        self.Extrinsic.clicked.connect(self.findExtrinsicPress)
        self.Distortion.clicked.connect(self.findDistortionPress)
        self.DisparityMap.clicked.connect(self.findDisparityMap)  
        self.drawAR.clicked.connect(self.findAR)         
        self.KeyPoint.clicked.connect(lambda: self.findKeyPoint(1))  
        self.MatchKeyPoint.clicked.connect(lambda: self.findKeyPoint(2))
        self.show()


#######-------------HOMEWORK 1.1--------------------##########
    def findCornerPress(self):
        print("Find Corner Button Pressed : \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            
            imS = cv2.resize(img, (1024, 1024))                    # Resize image
            cv2.imshow("output", imS)                            # Show image
            cv2.waitKey(0)                                      # Display the image infinitely until any keypress

        cv2.destroyAllWindows()    


#######-------------HOMEWORK 1.2--------------------##########
    def findIntrinsicPress(self):
        # Defining the dimensions of checkerboard
        print("Find Intrinsic Button Pressed : \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)        
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Camera matrix : \n")
        print(mtx)


#######-------------HOMEWORK 1.3--------------------##########
    def findExtrinsicPress(self):
        # Defining the dimensions of checkerboard
        print("Find Extrinsic Button Pressed : \n")
        a = self.ExtrinsicBox.currentIndex()

        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)        
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        R_matrix, _ = cv2.Rodrigues(rvecs[a])
        Rt_matrix = np.concatenate((R_matrix, tvecs[a]), axis=1)

        print("R_Matrix : \n")
        print(R_matrix)
        print("Rt_Matrix : \n")
        print(Rt_matrix)


#######-------------HOMEWORK 1.4--------------------##########
    def findDistortionPress(self):
        # Defining the dimensions of checkerboard
        print("Find Intrinsic Button Pressed : \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)        
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("dist : \n")
        print(dist)


#######-------------HOMEWORK2--------------------##########
    def draw(img, corners, imgpts):
        
        corner = tuple(corners[12].ravel())#12 16 58
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)

        corner = tuple(corners[16].ravel())#12 16 58
        # img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)

        corner = tuple(corners[58].ravel())#12 16 58
        # img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        # img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

    def findAR(self):
        print("Drawing Tetrahedron : \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        axis = np.float32([[5,1,0], [3,5,0], [3,3,-3],])
        
        for fname in glob.glob('Q2_Image/*.bmp'):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.

                corners2 = cv2.cornerSubPix(gray, corners, CHECKERBOARD,(-1,-1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.destroyAllWindows()    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        for fname in glob.glob('Q2_Image/*.bmp'):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD ,None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,CHECKERBOARD,(-1,-1),criteria)
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                img = Ui.draw(img,corners2,imgpts)
                imS = cv2.resize(img, (1024, 1024))                    # Resize image
                cv2.imshow('img',imS)
                k = cv2.waitKey(0) & 0xFF
                if k == ord('s'):
                    cv2.imwrite(fname[:6]+'.bmp', imS)
        cv2.destroyAllWindows()   


#######-------------HOMEWORK3--------------------##########
    def findDisparityMap(self):

        imgL = cv2.imread('Q3_Image/imL.png',0)
        imgR = cv2.imread('Q3_Image/imR.png',0)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL,imgR)
        plt.imshow(disparity,'gray')
        plt.show()  


#######-------------HOMEWORK4--------------------##########
    def findKeyPoint(self,i):
        # read images
        img1 = cv2.imread('Q4_Image/Aerial1.jpg')  
        img2 = cv2.imread('Q4_Image/Aerial2.jpg') 

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #sift
        sift = cv2.xfeatures2d.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(descriptors_1,descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)

        if i == 1 :
            img1_idx = []
            img2_idx = []
            for i in range(400,406):
                img1_idx.append(keypoints_1[matches[i].queryIdx])
                img2_idx.append(keypoints_2[matches[i].trainIdx])

            #draw keypoint
            imgN1 = cv2.drawKeypoints(img1, keypoints=img1_idx, outImage=img1, color= (0, 0, 255), flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            imgN2 = cv2.drawKeypoints(img2, keypoints=img2_idx, outImage=img2, color= (0, 0, 255), flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

            numpy_horizontal_concat = np.concatenate((imgN1, imgN2), axis=1)
            cv2.imshow('image', numpy_horizontal_concat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else :
            img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[400:406], img2, flags=2)
            cv2.imshow('image', img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()        
        
app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

 