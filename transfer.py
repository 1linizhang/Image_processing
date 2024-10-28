import cv2
import numpy as np

image=cv2.imread('loopy.jpg')

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
y=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

gra=cv2.magnitude(x,y)
gra=cv2.convertScaleAbs(gra)

_,edge1=cv2.threshold(gra,50,255,cv2.THRESH_BINARY)
HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

lo=np.array([100,50,50])
up=np.array([140,255,255])

blue1=cv2.inRange(HSV,lo,up)
white1=np.ones_like(image)*255
black1=cv2.bitwise_not(blue1)

loopy=cv2.bitwise_and(image,image,mask=black1)
back=cv2.bitwise_and(white1,white1,mask=blue1)
combine=cv2.add(loopy,back)

edge1= cv2.resize(edge1, (combine.shape[1], combine.shape[0]))
edge2= cv2.cvtColor(edge1, cv2.COLOR_GRAY2BGR)

result=cv2.addWeighted(combine,0.7,edge2,0.3,0)

cv2.imwrite('result.jpg',result)

cv2.imshow('Result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()