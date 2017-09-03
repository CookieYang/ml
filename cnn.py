import cv2
import numpy as np

image = cv2.imread(r'D:\datas\source\test1.jpg')
cv2.imshow('original',image)

# kernel = np.array([[.1,.1,.1],[.1,.1,.1],[.1,.1,.1]])
# rect = cv2.filter2D(image,-1,kernel)
# cv2.imwrite('rect.jpg',rect)

# kernel = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273.0
# gasussian = cv2.filter2D(image, -1, kernel)
# cv2.imwrite('guassian.jpg',gasussian)

# kernel = np.array([[0,-2,0],[-2,9,-2],[0,-2,0]])
# sharpen = cv2.filter2D(image,-1,kernel)
# cv2.imwrite('sharpen.jpg',sharpen)

# kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# edges = cv2.filter2D(image,-1,kernel)
# cv2.imwrite('edges.jpg',edges)

kernel = np.array([[-2,-2,-2,-2,0],[-2,-2,-2,0,2],[-2,-2,0,2,2],[-2,0,2,2,2],[0,2,2,2,2]])
emboss = cv2.filter2D(image,-1,kernel)
emboss = cv2.cvtColor(emboss,cv2.COLOR_BGR2GRAY)
cv2.imwrite('emboss.jpg',emboss)
# cv2.waitKey(0)