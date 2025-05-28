color_points = [[147,143],[146,10],[128,11],[303,141],[238,12]]
depth_points = [[193,144],[192,97],[180,96],[288,148],[240,98]]

import cv2
import numpy as np

# Load your two images
img1 = cv2.imread('color.png')  # The image to transform
img2 = cv2.imread('depth.png')  # The reference image

# Replace these with your actual matched points (at least 4)
pts_img1 = np.array(color_points, dtype=np.float32)

pts_img2 = np.array(depth_points, dtype=np.float32)

# Compute the homography matrix
H, status = cv2.findHomography(pts_img2,pts_img1, method=cv2.RANSAC)

# Warp the entire image1 to the viewpoint of image2
height, width, channels = img2.shape
warped_img1 = cv2.warpPerspective(img1, H, (width, height))

H_saved = np.array([[ 4.67834689e-01, -1.28873172e-01,  1.16438518e+02,0],
 [-5.65045485e-03,  2.39912243e-01,  9.10999747e+01,0],
 [-2.07257063e-04, -7.36308516e-04,  1.00000000e+00,0],
 [0,0,0,1]])

print(H)

for i in range(0,img2.shape[0],16):
    for j in range(0,img2.shape[1],16):
        res = H@np.array([i,320-j,1])
        res = res/res[2]
        cv2.circle(warped_img1,(int(res[0]),int(res[1])),1,(0,255,0))
        

# Save or display result
# cv2.imwrite('warped_image1.jpg', warped_img1)
# Or view with OpenCV (optional)
cv2.imshow('Warped Image', warped_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()