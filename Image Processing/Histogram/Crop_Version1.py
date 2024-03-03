import glob
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
import os
import numpy 
from PIL import Image, ImageTransform
from pathlib import Path
#FOR ALL DATA AT THE CAR FOLDER:
files = glob.glob(r"data\PLANE/*.txt")
images = glob.glob(r"data\PLANE/*.png")
i = 0
for file, image in zip(files, images):

    file    = open(file,"r")
    content = file.read().splitlines()
    data    = [list(map(float,line.split('\t')[0:13])) for line in content]

    file.close()
    #get all data individual 
    x_s = [[round(i) for i in x[0:7:2]] for x in data]
    y_s = [[round(i) for i in y[1:8:2]] for y in data]
    theta_s = [theta[8] for theta in data]
    lx_s = [lx[9] for lx in data]
    ly_s = [ly[10] for ly in data]
    width_s = [width[11] for width in data]
    height_s = [height[12] for height in data]

    #Read image using Opencv
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    #img = Image.open(image).convert('RGB')
    
    for x, y, theta, width, height, lx, ly in zip(x_s, y_s, theta_s, width_s, height_s, lx_s, ly_s):
        
        top_left_x = min(x)
        top_left_y = min(y)
        bot_right_x = max(x)
        bot_right_y = max(y)
        firstPoint = (top_left_x, top_left_y)
        #firstPoint = (((Rotated_Points[0]))).astype(int)
        #last point 
        endPoint = (bot_right_x, bot_right_y)
        color = (255, 0, 0)
        #result= image[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
        #result = cv2.rectangle(img, firstPoint, endPoint,color)
        #cnts   =np.array([[x[1], y[1]], [x[1], y[1]], [x[1], y[1]], [x[1], y[1]]], dtype=np.int32)
        #cv2.drawContours(img, [cnts], idx, (0,255,0), 3)
        #idx +=1
        #result= img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
        result= img[top_left_y:bot_right_y, top_left_x:bot_right_x]

        width, height = result.shape[:2]
        # print(f"height {height} and width {width}")
        x = height / width
        if x >1 :
            swapped_image = cv2.transpose(result)
            cv2.imwrite(f'{Path(__file__).parent}/data_rotated/image_{i:05d}.png',swapped_image)
            i +=1
        else:
            cv2.imwrite(f'{Path(__file__).parent}/data_rotated/image_{i:05d}.png',result)
            i +=1
        #print(f"shape of result is {result.shape}")


        #rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), theta, 1.0)

        # Apply the rotation to the image
        #rotated_image = cv2.warpAffine(result, rotation_matrix, (width, height))




        

        
    #cv2.imshow("Image",result)
    #idx +=1
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

rectangle_center = tuple((lx, ly))
rect = (rectangle_center, (width,height), (theta))
box = cv2.boxPoints(rect)
box = numpy.int0(box)
cv2.drawContours(result,[box],0, color=(255,0,0), thickness=2)



        transform=[x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3]]

        w =  ((transform[2]-transform[0])**2+(transform[3]-transform[1])**2)**0.5 

        h =  ((transform[4]-transform[2])**2+(transform[5]-transform[3])**2)**0.5
        result = img.transform((int(w),int(h)), ImageTransform.QuadTransform(transform))


        print(w,h)
"""