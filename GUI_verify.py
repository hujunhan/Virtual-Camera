import cv2
import numpy as np
import time
from virtual_camera import VirtualCamera 
from utility import FeatureMesh, nothing

WINDOW_NAME = 'img'

W = 2048
H = 2048
focal = 1024*3.45e-6
# h = 1.1 -0.3 # height of the camera (respect to the circle)
h = 0.8



gazebo = cv2.imread('image/test1.png', 0)  
X=0.1
Y=0.1
beta = -(np.pi/2-1.02)
alpha=0

# Image captured from gazebo camera. uncomment the following code to see different situation
# gazebo = cv2.imread('image/test2.png', 0) 
# X = 2.561412-2.2
# Y = -0.205481
# beta = -(np.pi/2-1.047197)
# alpha = 0.52359877
cam = VirtualCamera(0, beta, alpha, 0, 0, h, focal, [W, H])

# print(cam.RT_camera_in_world)

img = np.zeros((1000, 1000, 1), dtype=np.uint8)
img[:, :] = 255
cv2.circle(img, (500, 500), 500, 0)

# cv2.imshow('img',img)
# cv2.waitKey(1)
mesh = FeatureMesh(X, Y, 0.14, 0.14, img)
pimg = mesh.project_image_to_world()

index = np.where(mesh.oneDimg == 0)
index = np.squeeze(index)
# print(len(index))
pimg = pimg[index, :]

timea = time.time()

pixels = cam.project_world_to_pixel(pimg)

pixels, index = cam.crop_pixel(pixels, index)
timeb = time.time()
# print(f'Cost {timeb-timea} seconds')
# print(f'num:{len(pixels)}')
newimg = np.zeros((W, H))
newimg[:, :] = 255

# print(f'XX:{max(pixels[:,0])-min(pixels[:,0])}')
# print(f'YY:{max(pixels[:,1])-min(pixels[:,1])}')
newimg[pixels[:, 0], pixels[:, 1]] = mesh.oneDimg[index]
newimg = newimg.astype(np.uint8)

add_image = cv2.addWeighted(newimg, 0.7, gazebo, 0.3, 0)
# cv2.imwrite('2a.png',add_image)
cv2.imshow('img', add_image)
cv2.waitKey(0)