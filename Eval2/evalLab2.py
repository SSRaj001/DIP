import cv2,sys,math
import numpy as np

def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

def padding(img):
    H,W,C = img.shape
    paddedImage = np.zeros((H+4,W+4,C))
    paddedImage[2:H+2,2:W+2,:C] = img
    paddedImage[2:H+2,0:2,:C]=img[:,0:1,:C]
    paddedImage[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]
    paddedImage[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]
    paddedImage[0:2,2:W+2,:C]=img[0:1,:,:C]
    paddedImage[0:2,0:2,:C]=img[0,0,:C]
    paddedImage[H+2:H+4,0:2,:C]=img[H-1,0,:C]
    paddedImage[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]
    paddedImage[0:2,W+2:W+4,:C]=img[0,W-1,:C]
    return paddedImage

def resizeBicubic(src,ratio,a):
    h,w,ch = src.shape

    src = padding(src)

    dH = math.floor(h*ratio)
    dW = math.floor(w*ratio)
    newImage = np.zeros((dH,dW,3))

    H = 1/ratio

    for c in range(ch):
        for j in range(dH):
            for i in range(dW):
                x,y = i*H+2,j*H+2

                X = math.floor(x)
                x1 = 1+x-X
                x2 = x-X
                x3 = X-x+1
                x4 = X-x+2

                Y = math.floor(y)
                y1 = 1+y-Y
                y2 = y-Y
                y3 = Y-y+1
                y4 = Y-y+2
                bicubicPixels = np.matrix([[src[int(y-y1),int(x-x1),c],src[int(y-y2),int(x-x1),c],src[int(y+y3),int(x-x1),c],src[int(y+y4),int(x-x1),c]],
                                            [src[int(y-y1),int(x-x2),c],src[int(y-y2),int(x-x2),c],src[int(y+y3),int(x-x2),c],src[int(y+y4),int(x-x2),c]],
                                            [src[int(y-y1),int(x+x3),c],src[int(y-y2),int(x+x3),c],src[int(y+y3),int(x+x3),c],src[int(y+y4),int(x+x3),c]],
                                            [src[int(y-y1),int(x+x4),c],src[int(y-y2),int(x+x4),c],src[int(y+y3),int(x+x4),c],src[int(y+y4),int(x+x4),c]]])
                mat_x = np.matrix([u(x1,a),u(x2,a),u(x3,a),u(x4,a)])
                mat_y = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])

                newImage[j,i,c] = np.dot(np.dot(mat_x,bicubicPixels),mat_y)
    return newImage


def bound(nx,ny,w,h):
    if nx < 0:
        nx = 0
    if ny < 0:
        ny = 0
    if nx >= w:
        nx = w-1
    if ny >= h:
        ny = h-1
    return nx,ny

def resizeBilinear(src,tx,ty):
    h,w,ch = src.shape

    hratio = h/ty
    wratio = w/tx

    newImage = np.zeros((ty,tx,ch),src.dtype)   

    for y in range(newImage.shape[0]):
        for x in range(newImage.shape[1]):
            for c in range(newImage.shape[2]): 
                ny = int(y*hratio+0.5)
                nx = int(x*wratio+0.5)
                y_ = int(ny)
                x_ = int(nx)
                xx,yy = bound(x_,y_,w,h)
                p1 = src[yy,xx,c]
                xx,yy = bound(x_,y_+1,w,h)
                p2 = src[yy,xx,c]
                xx,yy = bound(x_+1,y_,w,h)
                p3 = src[yy,xx,c]
                xx,yy = bound(x_+1,y_+1,w,h)
                p4 = src[yy,xx,c]
                xf = nx-x_
                yf = ny-y_

                b = xf*p2 + (1 - xf)*p1
                t = xf*p4 + (1 - xf)*p3
                newImage[y,x,c] = int(max(0,(yf*t+(1-yf)*b)))
    return newImage

def resizeNearest(src,tx,ty):
    h,w,ch = src.shape

    hratio = h/ty
    wratio = w/tx

    newImage = np.zeros((ty,tx,ch),src.dtype)   

    for y in range(newImage.shape[0]):
        for x in range(newImage.shape[1]):
            for c in range(newImage.shape[2]): 
                ny = int(y*hratio+0.5)
                nx = int(x*wratio+0.5)
                nx,ny = bound(nx,ny,w,h)
                newImage[y,x,c] = src[ny,nx,c]
    return newImage        

if __name__ == "__main__":

    image = cv2.imread(sys.argv[1])

    nearestInterpolation = resizeNearest(image,1536,1152)
    bilinearInterpolation = resizeBilinear(image,1536,1152)
    bicubicInterpolation = resizeBicubic(image,2,-1/2)
    cv2.imwrite("nearestInterpolation.jpg",nearestInterpolation)
    cv2.imwrite("bilinearInterpolation.jpg",bilinearInterpolation)
    cv2.imwrite("bicubicInterpolation.jpg",bicubicInterpolation)
