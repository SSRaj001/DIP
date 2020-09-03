import cv2,sys
import numpy as np

def threshold(origImage,limit,maxValue):
    h,w,c = origImage.shape
    newArray = np.zeros((h,w,c))
    for i in range(0,h):
        for j in range(0,w):
            for k in range(0,c):
                if(origImage[i][j][k] > limit):
                    newArray[i][j][k] = maxValue
                else:
                   newArray[i][j][k] = 0
    return newArray

if __name__ == "__main__":
    path = sys.argv[1]

    image = cv2.imread(path)

    if image is None:
        print("Invalid Path/File")

    print(image.shape)
    
    limit = int(input("Enter Limit above which Max is applied : "))
    maxVal = int(input("Enter a Max Threshold Value : "))

    newImage = threshold(image,limit,maxVal)
    
    cv2.imshow("Input",image)
    cv2.imshow("Output",newImage)
    cv2.imwrite("OutImage.jpg",newImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()