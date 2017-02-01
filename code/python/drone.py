import cv2
import numpy as np
import os

def nothing(x):
    pass

def testwrite(img, pfx, view, i):

    oname = "/home/joaquin/bees/out/{}{}{}.JPG".format(pfx,view,i)
    retval=cv2.imwrite(oname,img)
    print "Wrote ", oname, retval

os.system("rm /home/joaquin/bees/out/*.JPG")
    
def cannyContours(mat, n):
    edges = cv2.Canny(mat,25,50,n)

    cv2.namedWindow('edges')
    cv2.createTrackbar('MinVal','edges',0,255,nothing)
    cv2.createTrackbar('MaxVal','edges',0,255,nothing)
    while(1):
        cv2.imshow('edges',edges)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

            # get current positions of four trackbars
        minval = cv2.getTrackbarPos('MinVal','edges')
        maxval = cv2.getTrackbarPos('MaxVal','edges')
            
        edges = cv2.Canny(mat,minval,maxval,n)

    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return edges, contours, minval, maxval

view='front'
for i in range(1,8):
    imname = "/home/joaquin/bees/data/{}{}.JPG".format(view,i)
    
    print "Img ", imname
    img = cv2.imread(imname)

    if img is not None:
        img = cv2.pyrDown(img)
        img = cv2.pyrDown(img)

        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        testwrite(h, "H", view, i)
        testwrite(s, "S", view, i)
        testwrite(v, "V", view, i)
        #img = cv2.pyrDown(img)
        #img = cv2.pyrDown(img)
        v=cv2.equalizeHist(v) 
        s=cv2.equalizeHist(s)
        h=cv2.equalizeHist(h)
        testwrite(h, "Hh", view, i)
        testwrite(v, "Vh", view, i)
        testwrite(s, "Sh", view, i)
        
        sigD=2
        n=5
        t=27
        ksize=(4*sigD+1,4*sigD+1)
        retval,s = cv2.threshold(s,t,255,cv2.THRESH_BINARY_INV)
        testwrite(s, "St", view, i)
        
#plot histograms
