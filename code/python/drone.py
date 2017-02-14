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
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        retval = beescv.testwrite(img0,'ex',num)
        time.sleep(2)
        
os.system("rm /home/jcasa/bees/out/*.JPG")

view='front'
exname = "/home/jcasa/bees/out/ex/ex55.JPG"
if cv2.imread(exname) is None:
    for i in range(1,8):
        imname = "/home/jcasa/bees/data/{}{}.JPG".format(view,i)

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

c64 = np.zeros([64*64,1])
c128 = np.zeros([128*128,1])

for i in range(0,56):
    num=i
    view = 'ex'
    imname = "/home/jcasa/bees/out/ex/{}{}.JPG".format(view,i)
    
    print "Img ", imname
    img = cv2.imread(imname)
    if img is not None:
        bwname = "/home/jcasa/bees/out/bw/{}{}.JPG".format('128bw',i)
        print bwname
        if cv2.imread(bwname) is None:
            
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
            ns = int(max(16, int(max(np.power(2,np.round(np.log2(gray.shape)))))))
            ns=np.array([ns,ns])
            change = ns-gs
            if change[0]<0:
                crop_top=int(abs(np.round(change[0]/2)))
                crop_bottom=int(abs(change[0]+crop_top))
                gray_crop = gray[crop_top:-crop_bottom,:]
                change[0]=0
            else:
                gray_crop = gray

            gs = np.array(gray_crop.shape)    
            
            if change[1]<0:
                crop_left=int(abs(np.round(change[1]/2)))
                crop_right=int(abs(change[1]+crop_left))
                gray_crop2 = gray_crop[:,crop_left:-crop_right]
                change[1]=0
            else:
                gray_crop2 = gray_crop

            gs = np.array(gray_crop2.shape)    
            
            top = int(np.round(change[0]/2))
            bottom = int(change[0]-top)

            left = int(np.round(change[1]/2))
            right = int(change[1]-left)

            gray_pad = cv2.copyMakeBorder(gray_crop2,top,bottom,left,right,borderType=cv2.BORDER_REPLICATE)
            gray_pad_pyr = cv2.pyrDown(gray_pad)
            pfx = str(max(gray_pad_pyr.shape)) + 'bw'
            retval = beescv.testwrite(gray_pad_pyr,pfx,i)
            os.system("mv /home/jcasa/bees/out/*.JPG /home/jcasa/bees/out/bw/")

    g64name = '/home/jcasa/bees/out/bw/64bw%s.JPG'%(i)
    g128name = '/home/jcasa/bees/out/bw/128bw%s.JPG'%(i)
    print g64name, g128name
    g64 = cv2.imread(g64name,cv2.IMREAD_GRAYSCALE)
    g128 = cv2.imread(g128name,cv2.IMREAD_GRAYSCALE)
    if g64 is not None:
        g64c = g64.reshape([-1,1])-np.mean(g64.reshape([-1,1]))
        g128c = g128.reshape([-1,1])-np.mean(g128.reshape([-1,1]))
        c64 = np.hstack((c64,g64c))
        c128 = np.hstack((c128,g128c))


c64=np.delete(c64,(0),axis=1)
c128=np.delete(c128,(0),axis=1)

eigenvectors, eigenvalues, V = np.linalg.svd(c64, full_matrices=False)
k=np.argmin(np.abs(.95-np.cumsum(eigenvalues)/np.sum(eigenvalues)))
e64 = eigenvectors[:,0:k]
p64 = np.dot(e64.T,c64)
im64 = e64.reshape([64,64,-1])
for i in range(0,k):
    immax = np.max(np.max(im64[:,:,i]))
    immin = np.min(np.min(im64[:,:,i]))
    im64[:,:,i]=beescv.normalize(im64[:,:,i])
    pfx = '64v'
    retval = beescv.testwrite(im64[:,:,i],pfx,i)
            
eigenvectors, eigenvalues, V = np.linalg.svd(c128, full_matrices=False)
k=np.argmin(np.abs(.95-np.cumsum(eigenvalues)/np.sum(eigenvalues)))
e128 = eigenvectors[:,0:k]
p128 = np.dot(e128.T,c128)
im128 = e128.reshape([128,128,-1])
for i in range(0,k):
    immax = np.max(np.max(im128[:,:,i]))
    immin = np.min(np.min(im128[:,:,i]))
    im128[:,:,i]=beescv.normalize(im128[:,:,i])
    pfx = '128v'
    retval = beescv.testwrite(im128[:,:,i],pfx,i)

os.system("mv /home/jcasa/bees/out/*.JPG /home/jcasa/bees/out/eigenbees/")

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

view='front'
for i in range(1,8):
    imname = "/home/jcasa/bees/data/{}{}.JPG".format(view,i)

    print "Img ", imname
    img = cv2.imread(imname)
        
    if img is not None:
        for meth in methods:
            
            imgmatch = img.copy()
            gray = cv2.cvtColor(imgmatch, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = np.float32(gray)
            method = eval(meth)
            for j in range(0,e64.shape[1]):
                # Apply template Matching
                template=np.float32(im64[:,:,j])
                res = cv2.matchTemplate(gray,template,method)
                pfx=meth+'_64_%s_'%(j)
                retval = beescv.testwrite(beescv.normalize(res),pfx,i)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                print min_val,max_val,min_loc,max_loc
            for j in range(0,e128.shape[1]):
                # Apply template Matching
                template=np.float32(im128[:,:,j])
                res = cv2.matchTemplate(gray,template,method)
                pfx=meth+'_128_%s_'%(j)
                retval = beescv.testwrite(beescv.normalize(res),pfx,i)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                print min_val,max_val,min_loc,max_loc

os.system("mv /home/jcasa/bees/out/*.JPG /home/jcasa/bees/out/tm/")
