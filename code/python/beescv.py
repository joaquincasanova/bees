import numpy as np
import cv2
import time
def hist_plots(img):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
    time.sleep(2)
    plt.close()
    #plot histograms

def testwrite(img, pfx, i):
    oname = "/home/jcasa/bees/out/{}{}.JPG".format(pfx,i)
    retval=cv2.imwrite(oname,img)
    print "Wrote ", oname, retval
    return retval

def nothing(x):
    pass

def normalize(x):
    y=(x-np.min(x))/(np.max(x)-np.min(x))*255.
    return y

def localSD(mat, n):
    
    mat=np.float32(mat)
    mu = cv2.blur(mat,(n,n))
    mdiff=mu-mat
    mat2=cv2.blur(np.float64(mdiff*mdiff),(n,n))
    sd = np.float32(cv2.sqrt(mat2))
    sdn=normalize(sd)

    return sdn

def readsplit(imname):
        
    img = cv2.imread(imname)
    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
    else:
        h = None
        s = None
        v = None
        img = None
    return h, s, v, img

def opening_adjust(mat):

    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    cv2.namedWindow('open')
    opening = cv2.morphologyEx(mat, cv2.MORPH_OPEN, kernel)
    cv2.createTrackbar('n','open',0,50,nothing)
    while(1):
        cv2.imshow('open',opening)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        n = cv2.getTrackbarPos('n','open')
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*n+1,2*n+1))
        opening = cv2.morphologyEx(mat, cv2.MORPH_OPEN, kernel)

    cv2.destroyAllWindows()

    return opening, n

def thresh_adjust(mat):
    
    thresh =128
    retval,out=cv2.threshold(mat, thresh, 255, cv2.THRESH_BINARY)

    cv2.namedWindow('thresh')
    cv2.createTrackbar('thresh','thresh',0,255,nothing)
    while(1):
        cv2.imshow('thresh',out)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        thresh = cv2.getTrackbarPos('thresh','thresh')
        retval,out=cv2.threshold(mat, thresh, 255, cv2.THRESH_BINARY)


    cv2.destroyAllWindows()

    return out,thresh

def bilat_adjust(mat):

    bilat = cv2.adaptiveBilateralFilter(mat, -1, 7, 7)

    cv2.namedWindow('bilat')
    cv2.createTrackbar('sigC','bilat',0,100,nothing)
    cv2.createTrackbar('sigD','bilat',0,100,nothing)
    while(1):
        cv2.imshow('bilat',bilat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        n = -1
        sigC = cv2.getTrackbarPos('sigC','bilat')
        sigD = cv2.getTrackbarPos('sigD','bilat')
        bilat = cv2.bilateralFilter(mat, n, sigC, sigD)

    cv2.destroyAllWindows()

    return bilat, sigC, sigD
    
def canny_contours(mat, n):
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

