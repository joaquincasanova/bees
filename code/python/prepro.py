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
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),col,2)
        retval = beescv.testwrite(img0[min(iy,y):max(iy,y),min(ix,x):max(ix,x)],'ex',num)
        time.sleep(2)
        num+=1
    elif event == cv2.EVENT_RBUTTONDOWN:
        retval = beescv.testwrite(img0,'ex',num)
        time.sleep(2)
        

view='front'
view='front'
for i in range(1,8):
    imname = "/home/jcasa/bees/data/{}{}.JPG".format(view,i)

    h,s,v,img=beescv.readsplit(imname)
    if img is not None:
        H = beescv.normalize(h)
        S = beescv.normalize(s)
        V = beescv.normalize(v)
        retval = beescv.testwrite(h,'H',i)
        retval = beescv.testwrite(s,'S',i)
        retval = beescv.testwrite(v,'V',i)
        os.system('mv /home/jcasa/out/*.JPG /home/jcasa/out/hsv/*.JPG')
        out, thresh = beescv.thresh_adjust(s)
        os.system('mv /home/jcasa/out/*.JPG /home/jcasa/out/thresh/*.JPG')
        retval = beescv.testwrite(out,'T',i)        
        #edges, contours, minval, maxval = beescv.canny_contours(s, 5)
        #retval = beescv.testwrite(edges,'E',i)
        #os.system('mv /home/jcasa/out/*.JPG /home/jcasa/out/canny/*.JPG')
if False:    
    for i in range(1,8):
        imname = "/home/jcasa/bees/data/{}{}{}.jpg".format(view,i,'crop')

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

    os.system("mv /home/jcasa/bees/out/*.JPG /home/jcasa/bees/out/ex/")
