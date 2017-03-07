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
    dr="/home/jcasa/bees/out/ex/"
    col = np.random.randint(0,255,3)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),col,2)
        retval = beescv.testwrite(dr,img0[min(iy,y):max(iy,y),min(ix,x):max(ix,x)],'ex',num)
        time.sleep(2)
        num+=1
    elif event == cv2.EVENT_RBUTTONDOWN:
        retval = beescv.testwrite(dr,img0,'ex',num)
        time.sleep(2)

view='front'
if False:
    for i in range(1,8):
        imname = "/home/jcasa/bees/data/{}{}.JPG".format(view,i)

        h,s,v,l,a,b,img=beescv.readsplit(imname)
        h=cv2.equalizeHist(h)
        s=cv2.equalizeHist(s)
        v=cv2.equalizeHist(v)
        l=cv2.equalizeHist(l)
        a=cv2.equalizeHist(a)
        b=cv2.equalizeHist(b)
        if img is not None:
            dr="/home/jcasa/bees/out/hsv/"
            retval = beescv.testwrite(dr,h,'H',i)
            retval = beescv.testwrite(dr,s,'S',i)
            retval = beescv.testwrite(dr,v,'V',i)
            retval = beescv.testwrite(dr,l,'L',i)
            retval = beescv.testwrite(dr,a,'a',i)
            retval = beescv.testwrite(dr,b,'b',i)
            #bilat, sigC, sigD =beescv.bilat_adjust(s)
            bilat = cv2.bilateralFilter(b, -1, 30, 20)
            dr="/home/jcasa/bees/out/bilat/"
            retval = beescv.testwrite(dr,bilat,'B_'+str(30)+'_'+str(20)+'_',i)

            out, thresh = beescv.thresh_adjust(bilat)
            #retval,out=cv2.threshold(bilat, thresh, 255, cv2.THRESH_BINARY)
            dr="/home/jcasa/bees/out/thresh/"

            retval = beescv.testwrite(dr,out,'T_'+str(thresh)+'_',i)        
            contours, hierarchy = cv2.findContours(out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            dr="/home/jcasa/bees/out/br/"
            x,y,w,h,imgRect,cntMax=beescv.findBR(img,contours)
            retval = beescv.testwrite(dr,imgRect,'T_BR',i)

            imgcontours=img.copy()
            cv2.drawContours(imgcontours,[cntMax],0,(0,255,0),5)
            dr="/home/jcasa/bees/out/c/"
            retval = beescv.testwrite(dr,imgcontours,'C',i)
        
if True:    
    for i in range(1,8):
        imname = "/home/jcasa/bees/out/br/T_BR{}.JPG".format(i)
        h,s,v,l,a,b,img=beescv.readsplit(imname)
        print "Img ", imname
        

        if img is not None:
            img0 = img.copy()
            
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',select_example)

            while(1):
                cv2.imshow('image',img0)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            cv2.destroyAllWindows()
