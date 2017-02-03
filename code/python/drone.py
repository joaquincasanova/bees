import cv2
import numpy as np
import os
import beescv
import time
from matplotlib import pyplot as plt

drawing = False # true if mouse is pressed
ix,iy = -1,-1
num = 0

# mouse callback function
def select_example(event,x,y,flags,param):
    global ix,iy,drawing,mode,num
    col = np.random.randint(0,255,3)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    #elif event == cv2.EVENT_MOUSEMOVE:
    #    if drawing == True:
    #        cv2.rectangle(img,(ix,iy),(x,y),col,2)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),col,2)
        retval = beescv.testwrite(img0[min(iy,y):max(iy,y),min(ix,x):max(ix,x)],'ex',num)
        time.sleep(2)
        if retval is True:num+=1

os.system("rm /home/jcasa/bees/out/*.JPG")

#view='front'
#for i in range(1,8):
#    imname = "/home/jcasa/bees/data/{}{}.JPG".format(view,i)
    
#    print "Img ", imname
#    img = cv2.imread(imname)
#    print img
    
#    if img is not None:
#        img0 = img.copy()
#        cv2.namedWindow('image')
#        cv2.setMouseCallback('image',select_example)

#        while(1):
#            cv2.imshow('image',img)
#            k = cv2.waitKey(1) & 0xFF
#            if k == 27:
#                break

#        cv2.destroyAllWindows()
os.system("mv /home/jcasa/bees/out/*.JPG /home/jcasa/bees/out/ex/")

for i in range(0,56):
    view = 'ex'
    imname = "/home/jcasa/bees/out/ex/{}{}.JPG".format(view,i)
    
    print "Img ", imname
    img = cv2.imread(imname)
    
    if img is not None:
        img0 = img.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',select_example)

        while(1):
            cv2.imshow('image',img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

    imname = "/home/jcasa/bees/out/{}{}.JPG".format(view,i)
    
    print "Img ", imname
    img_crop = cv2.imread(imname)
        # Convert BGR to HSV
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    gs = np.array(gray.shape)    
    print "shape: ", gs
    ns = int(max(np.power(2,np.round(np.log2(gray.shape)))))
    ns=np.array([ns,ns])
    change = ns-gs
    print change
    if change[0]<0:
        crop_top=int(abs(np.round(change[0]/2)))
        crop_bottom=int(abs(change[0]+crop_top))
        gray_crop = gray[crop_top:-crop_bottom,:]
        change[0]=0
    else:
        gray_crop = gray

    gs = np.array(gray_crop.shape)    
    print "shape: ", gs
        
    if change[1]<0:
        crop_left=int(abs(np.round(change[1]/2)))
        crop_right=int(abs(change[1]+crop_left))
        gray_crop2 = gray_crop[:,crop_left:-crop_right]
        change[1]=0
    else:
        gray_crop2 = gray_crop

    gs = np.array(gray_crop2.shape)    
    print "shape: ", gs

    top = int(np.round(change[0]/2))
    bottom = int(change[0]-top)
    
    left = int(np.round(change[1]/2))
    right = int(change[1]-left)

    gray_pad = cv2.copyMakeBorder(gray_crop2,top,bottom,left,right,borderType=cv2.BORDER_REPLICATE)

    
    while(1):
        cv2.imshow("gray",gray_pad)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    print "new shape: ", gray_pad.shape
    
    gray_pad_up = cv2.pyrUp
    gray_pad_down = cv2.pyrDown
