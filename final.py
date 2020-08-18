# -*- coding: utf-8 -*-
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from skimage import io, data_dir, transform, color
import os
from glob import glob
import math

MAX_MATCHES = 910




#convert to grayscale 
#gray1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2BGR)
#gray2 = cv2.cvtColor(crop2, cv2.COLOR_RGB2BGR)
#use Gaussian
#Gauss1 = cv2.GaussianBlur(gray1,ksize = (3, 3), sigmaX = 0)
#Gauss2 = cv2.GaussianBlur(gray2,ksize = (3, 3), sigmaX = 0)
#use Canny edge detetctor
#edges1 = cv2. Canny(Gauss1, 20, 30)
#edges2 = cv2. Canny(Gauss2, 20, 30)

# 1. using keypoint detectors implemented in OpenCV (SIFT, SURF, ORB) 
#Find keypoints using ORB
     # detector locator (where the points are), descriptor (an array of numbers, recognition)

# 2. homography:
# use 3*3 matrix to despribe the relation between two images
# findHomography : using four points
#                : h, status = cv2. findHomography(point1, point2)


def auto_canny(image):

  blocksize = 21;
  constbvalue = 0;
  maxVal = 255;
  '''
  ADAPTIVE_THRESH_MEAN_C = 0
  ADAPTIVE_THRESH_GAUSSIAN_C = 1

  THRESH_BINARY = 0
  THRESH_BINARY_INV = 1

  '''
  adaptiveMethod = 1;
  thresholdType = 1;
  #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 33, 0)
  image = cv2.adaptiveThreshold(image, maxVal, adaptiveMethod, thresholdType, blocksize, constbvalue)

  return image

def cal_angle(x_point1, y_point1, x_point2, y_point2):
  angle = 0
  de_y = y_point1 - y_point2
  de_x = x_point1 - x_point2
  de_x = 343 - de_x
  #angle = math.atan2(de_y, de_x)
  angle = de_y/de_x


  return angle


def alignImages(im1, im2,imagename):


  #convert to grayscale 
  edges1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  edges2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  #print(edges1[100,100])
  #print(edges2[100,100])

  #use Gaussian
  
  #edges1 = cv2.GaussianBlur(im1,ksize = (3, 3), sigmaX = 0)

  #edges2 = cv2.GaussianBlur(im2,ksize = (3, 3), sigmaX = 0)
  #use Canny edge detetctor
  #lower = 0
  #max_lower = 100

  #edges1 = auto_canny(edges1)
  #edges2 = auto_canny(edges2)

  orb = cv2.ORB_create(MAX_MATCHES)
  #orb = cv2.xfeatures2d.SIRF_create(float(MAX_MATCHES))
  keypoints1, descriptors1 = orb.detectAndCompute(edges1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(edges2, None)



  # Match features.
  '''
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params,search_params)
  matches = flann.knnMatch(descriptors1, descriptors2, k = 2)
  '''

  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
  matches = matcher.match(descriptors1, descriptors2)
  
  #matches = matcher.knnMatch(descriptors1, descriptors2, k = 1)
  
  # Sort matches by score
  #matches = [m for (m,n) in matches if m.distance < 0.75*n.distance]
  

  matches.sort(key=lambda x: x.distance)
  
  # Remove not so good matches
  #numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  #matches = matches[:numGoodMatches]

  j = 0
  for i in range(len(matches)):
    

    if (matches[i].distance < 50):


      j = j+1
      if (j > 80):
        i = 1500


  if (j > 10):
    m = 1
    matches.sort(key=lambda x: x.distance)

    matches = matches[:j]
    

    # Draw top matches
    #imMatches = cv2.drawMatches(edges1,keypoints1, edges2, keypoints2, matches, None, flags = 2)
    #cv2.imwrite("matches2.jpg", imMatches)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)
    mat = np.zeros((len(matches), 2), dtype = np.float32)
    #angle_p = np.zeros(1, dtype = np.float32)
    #mat = np.zeros(len(matches), dtype = np.float32)
    for i, match in enumerate(matches):
      #print(i)
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt

      mat[i, :1] = cal_angle(points1[i,:1],points1[i,1:2],points2[i,:1],points2[i,1:2])
    
    max_val = 0
    angle = 0.0
    tore = 0.0


    # cal the frequency
    for a in range(len(matches)):
      for b in range(len(matches)):
        #if (abs(round(float(mat[a, :1]),5) - round(float(mat[b, :1]),5)) < abs(tore)):
        if (round(float(mat[a, :1]),1) == round(float(mat[b, :1]),1)):
          mat[a, 1:2] = mat[a, 1:2] + 1
      # get the max value
      if (int(mat[a, 1:2]) > max_val):
        max_val = int(mat[a, 1:2])
        angle = round(float(mat[a, :1]),5)
        #tore = angle * 0.2
    tore = 100 * 0.2

    # filter values

    i = 0
    for n in range(len(matches)):
      if (abs(round(float(mat[n, :1]),5) - angle) < abs(tore)):
        quit = 0
        #print(round(float(mat[n, :1]),5) - angle)

      else:
 
        matches.remove(matches[n-i])
        i = i+1

    


    if (len(matches) > 4):

      for i, match in enumerate(matches):
        #print(i)
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
      '''
      imMatches = cv2.drawMatches(edges1,keypoints1, edges2, keypoints2, matches, None, flags = 2)
      cv2.imwrite("matches2.jpg", imMatches)
      '''




      # Find homography
      h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
      
      
      height, width, channels = im2.shape

        
      if h is None:
        im1Reg = im1
        m = 0

        #print imagename[:-4],",",h[0][0]
      else:
        #print('1')
        
        if (h[0][0] > 0.68 and h[0][0] < 1.3 and 
          h[0][1] > -0.53 and h[0][1] < 0.57 and 
          h[0][2] > -54 and h[0][2] < 93 and 
          h[1][0] > -0.121 and h[1][0] < 0.19 and 
          h[1][1] > 0.7 and h[1][1] < 1.63 and 
          h[1][2] > -75 and h[1][2] < 40 and 
          h[2][0] > -0.00085 and h[2][0] < 0.00092 and
          h[2][1] > -0.0009 and h[2][1] <0.0013
        ):
      
        # Use homography
          height, width, channels = im2.shape
          im1Reg = cv2.warpPerspective(im1, h, (width, height))
        
        else:
          im1Reg = im1
          m = 0
        
          #im1Reg = cv2.warpPerspective(im1, h, (width, height))
        #print imagename[:-4],",",h[2][1]
      #print(imagename)

      '''
      count = 0
      for i in range(len(mask)):
        if (mask[i] == 0):
          count = count + 1
      count = count / 1.0
      accuracy = (len(mask)-count)/len(mask)
      '''


      '''
      rows, cols = np.where(im1Reg[:,:,0] !=0)
      min_row, max_row = min(rows), max(rows) +1
      min_col, max_col = min(cols), max(cols) +1
      im1Reg = im1Reg[min_row:max_row,min_col:max_col,:]
      print(max_row - min_row)
      print(max_col - min_col)
      '''
    else:
      im1Reg = im1
      m = 0
    
    
  else: 
      im1Reg = im1
      m = 0
  return im1Reg, m



def img2video(img_dir, save_name, img_size):

  fps = 5
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  video_writer = cv2.VideoWriter(save_name, fourcc, fps, img_size)
  
  coll = io.ImageCollection(img_dir + '.jpg')
  for i in range (len(coll)):
    imgname = os.path.join(img_dir,str(i)+'.jpg')
    frame = cv2.imread(imgname)
    video_writer.write(frame)
  print("done")
  #video_writer.release()


def img_name(path, list = {}):
  path = os.path.expanduser(path)
  i = 0
  for f in os.listdir(path):
    if f.lower().endswith('.jpg'):
      #print f.strip()
      list[i] = f.strip()
      i = i + 1

  return list




if __name__ == '__main__':
  jpg_path = '/Users/shuqiwu/Desktop/0401' 
  save_path = '/Users/shuqiwu/Desktop/right'
  save_path2 = '/Users/shuqiwu/Desktop/wrong'

  #coll = io.ImageCollection(jpg_path + '/*.jpg')  # see how many images in the file
  list_l = {}
  img_name(jpg_path, list_l)
  #print (list[0])

  #1312374193.jpg

  #list_l[0] = '1312374193.jpg'
  #ref_image = cv2.imread(jpg_path + '/' + '1312374193.jpg')  
  #ref_image = ref_image[22:288,5:348] 
  
  for i in range(len(list_l)):
    #print(i)
    print(list_l[i])
  
   
    if i == 0:
      #ref_image = cv2.imread('/Users/shuqiwu/Documents/project_code/04/%d.jpg' %i)
      #ref_image = cv2.cvtColor(coll[i], cv2.COLOR_RGB2BGR)
      ref_image = cv2.imread(jpg_path + '/' + list_l[i])
     
      ref_image = ref_image[22:288,5:348]
      #ref_image2 = ref_image[3:260,15:340]
      #print (i)
      
      #h, w, c = ref_image.shape
      cv2.imwrite(save_path + '/' + list_l[i], ref_image/2 + ref_image/2)
    else:
   
      #imaging = cv2.imread('/Users/shuqiwu/Documents/project_code/04/%d.jpg' %i)

      #imaging = coll[i]
      #imaging = cv2.cvtColor(coll[i], cv2.COLOR_RGB2BGR)
      imaging = cv2.imread(jpg_path + '/' + list_l[i])
      
      imaging2 = imaging[22:288,5:348]

      
      imReg, m= alignImages(imaging2, ref_image,list_l[i])
      #image_w = imReg/2 + ref_image/2
      if m == 0: 
        cv2.imwrite(save_path2+ '/' + list_l[i], imaging)

      else: 

        #imReg = imReg[8:250,8:300]
        image_w = imReg
        #image_w2 = imReg[3:260,15:340]
        #print (i)
        #imReg = cv2.cvtColor(imReg, cv2.COLOR_RGB2GRAY)


        #cv2.imwrite('/Users/shuqiwu/Documents/project_code/01a/%d.jpg' %i, image_w2)
        cv2.imwrite(save_path + '/' + list_l[i], image_w/2+ref_image/2)
        #img_root = '/Users/shuqiwu/Documents/project_code/01a/'
 
        #img2video(img_root, "a.avi",img_size = (257,325))  

        #print(imReg[1,1]);

      '''
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        imaging = cv2.cvtColor(imaging, cv2.COLOR_BGR2RGB)
        image_w = cv2.cvtColor(image_w, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 1, 1), plt.imshow((ref_image/2+image_w/2))
        plt.title("aligned"), plt.xticks([]), plt.yticks([])

        plt.subplot(2, 1, 2), plt.imshow((ref_image/2+imaging/2))
        plt.title("no aligned"), plt.xticks([]), plt.yticks([])
  
        plt.show()
      '''

  
  

  
        #print(len(coll))
 

      #plt.show()


