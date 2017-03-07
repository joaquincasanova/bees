import cv2
import numpy as np
import os
import beescv
import time
from matplotlib import pyplot as plt
   
view='front'
    
c16 = np.zeros([16*16,1])
c32 = np.zeros([32*32,1])  
c64 = np.zeros([64*64,1])
c128 = np.zeros([128*128,1])
c256 = np.zeros([256*256,1])
c = [c16,c32,c64,c128,c256]
   
for i in range(0,56):
    view = 'ex'
    imname = "/home/jcasa/bees/out/ex/{}{}.JPG".format(view,i)
    
    print "Img ", imname
    h,s,v,l,a,b,img=beescv.readsplit(imname)
    
    if img is not None:

        gray = cv2.equalizeHist(cv2.bilateralFilter(b,-1,7,7))

        gs = np.array(gray.shape)    
        ns = int(min(256,max(16, int(max(np.power(2,np.round(np.log2(gray.shape))))))))
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
        pfx = str(max(gray_pad.shape)) + 'bw'
        dr='/home/jcasa/bees/out/bw/'
        retval = beescv.testwrite(dr,gray_pad,pfx,i)

        g16name = '/home/jcasa/bees/out/bw/16bw%s.JPG'%(i)
        g32name = '/home/jcasa/bees/out/bw/32bw%s.JPG'%(i)
        g64name = '/home/jcasa/bees/out/bw/64bw%s.JPG'%(i)
        g128name = '/home/jcasa/bees/out/bw/128bw%s.JPG'%(i)
        g256name = '/home/jcasa/bees/out/bw/256bw%s.JPG'%(i)
        g16 = cv2.imread(g16name,cv2.IMREAD_GRAYSCALE)
        g32 = cv2.imread(g32name,cv2.IMREAD_GRAYSCALE)
        g64 = cv2.imread(g64name,cv2.IMREAD_GRAYSCALE)
        g128 = cv2.imread(g128name,cv2.IMREAD_GRAYSCALE)
        g256 = cv2.imread(g256name,cv2.IMREAD_GRAYSCALE)

        g = [g16,g32,g64,g128,g256]
        gn =[g16name,g32name,g64name,g128name,g256name]
        k=0
        for ig in g:
            k+=1
            if ig is not None:
                g2k = np.int(np.power(2,k+3))
                print "Original is: ", g2k
                break
        if k==1:
            g32 = cv2.pyrUp(g16)
            g64 = cv2.pyrUp(g32)
            g128 = cv2.pyrUp(g64)
            g256 = cv2.pyrUp(g128)
        elif k==2:
            g16 = cv2.pyrDown(g32)
            g64 = cv2.pyrUp(g32)
            g128 = cv2.pyrUp(g64)
            g256 = cv2.pyrUp(g128)
        elif k==3:
            g32 = cv2.pyrDown(g64)
            g16 = cv2.pyrDown(g32)
            g128 = cv2.pyrUp(g64)
            g256 = cv2.pyrUp(g128)
        elif k==4:
            g64 = cv2.pyrDown(g128)
            g32 = cv2.pyrDown(g64)
            g16 = cv2.pyrDown(g32)
            g256 = cv2.pyrUp(g128)
        elif k==5:
            g128 = cv2.pyrDown(g256)
            g64 = cv2.pyrDown(g128)
            g32 = cv2.pyrDown(g64)
            g16 = cv2.pyrDown(g32)
            
                    
        g = [g16,g32,g64,g128,g256]
        gc = [None,None,None,None,None]
        k=0
        for ig in g:
            print gn[k], cv2.imwrite(gn[k],ig)
            gc[k] = ig.reshape([-1,1])-np.mean(ig.reshape([-1,1]))
            c[k] = np.hstack((c[k],gc[k]))
            k+=1
k=0
e = [None,None,None,None,None]
p = [None,None,None,None,None]
im = [None,None,None,None,None]

for ic in c:
    c[k]=np.delete(ic,(0),axis=1)    
    eigenvectors, eigenvalues, V = np.linalg.svd(c[k], full_matrices=False)
    kk=np.argmin(np.abs(.95-np.cumsum(eigenvalues)/np.sum(eigenvalues)))
    e[k] = eigenvectors[:,0:kk]
    p[k] = np.dot(e[k].T,c[k])
    g2k = np.int(np.power(2,k+4))
    im[k] = e[k].reshape([g2k,g2k,-1])
    for i in range(0,kk):
        im[k][:,:,i]=beescv.normalize(im[k][:,:,i])
        pfx = str(g2k)+'v'
        dr='/home/jcasa/bees/out/eigenbees/'
        retval = beescv.testwrite(dr,im[k][:,:,i],pfx,i)
    k+=1

methods = ['cv2.TM_SQDIFF_NORMED']

view='front'
for i in range(1,8):
    imname = "/home/jcasa/bees/data/{}{}{}.jpg".format(view,i,'crop')


    print "Img ", imname
    h,s,v,l,a,b,img=beescv.readsplit(imname)
    if img is not None:
        for meth in methods:          
            imgmatch = b.copy()
            b=cv2.equalizwHist(b)
            bilat = cv2.bilateralFilter(b, -1, 30, 20)
            gray = np.float32(bilat)
            method = eval(meth)
            for k in range(0,5):
                for j in range(0,e[k].shape[1]):
                    # Apply template Matching
                    template=np.float32(im[k][:,:,j])
                    res = cv2.matchTemplate(gray,template,method)
                    pfx=meth+'_scale_%s_eig_%s_img_'%(np.power(2,k+4),j)
                    dr='/home/jcasa/bees/out/tm/'
                    retval = beescv.testwrite(dr,beescv.normalize(res),pfx,i)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    print min_loc
