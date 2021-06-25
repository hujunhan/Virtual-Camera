import cv2
import numpy as np
import time
from virtual_camera import VirtualCamera 
from utility import FeatureMesh, nothing

WINDOW_NAME = 'Playground'
cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("X(-1m 1m)", WINDOW_NAME, 110, 200, nothing,)
cv2.createTrackbar("Y(-1m 1m)", WINDOW_NAME, 110, 200, nothing)
cv2.createTrackbar("beta(-30 30)", WINDOW_NAME, 30, 60, nothing)
cv2.createTrackbar("h", WINDOW_NAME, 800, 1000, nothing)
W = 2048
H = 2048
focal = 1024*3.45e-6
h = 0.8
gazebo = cv2.imread('image/test1.png', 0)

while True:
    X = -(cv2.getTrackbarPos("X(-1m 1m)", WINDOW_NAME)-100) / 100
    Y = (cv2.getTrackbarPos("Y(-1m 1m)", WINDOW_NAME) - 100)/100
    beta = -(cv2.getTrackbarPos("beta(-30 30)",WINDOW_NAME)-30)/180*np.pi
    h=(cv2.getTrackbarPos("h", WINDOW_NAME) )/1000
    
    
    cam = VirtualCamera(0, beta, 0, 0, 0, h, focal, [W, H])

    # print(cam.RT_camera_in_world)

    img = np.zeros((1000, 1000, 1), dtype=np.uint8)
    img[:, :] = 255
    cv2.circle(img, (500, 500), 500, 0)

    mesh = FeatureMesh(X, Y, 0.14, 0.14, img)
    pimg = mesh.project_image_to_world()

    index = np.where(mesh.oneDimg == 0)
    index = np.squeeze(index)

    pimg = pimg[index, :]

    timea = time.time()

    pixels = cam.project_world_to_pixel(pimg)

    pixels, index = cam.crop_pixel(pixels, index)
    timeb = time.time()

    newimg = np.zeros((W, H))
    newimg[:, :] = 255

    newimg[pixels[:, 0], pixels[:, 1]] = mesh.oneDimg[index]
    newimg = newimg.astype(np.uint8)

    add_image = cv2.addWeighted(newimg, 0.7, gazebo, 0.3, 0)
    cv2.imshow(WINDOW_NAME, add_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break