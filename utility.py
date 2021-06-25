import numpy as np

class FeatureMesh:
    def __init__(self, x, y, width, height, img) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.img = img
        self.oneDimg = img.reshape(-1)

    def project_image_to_world(self):
        x = np.linspace(self.x-self.width/2, self.x +
                        self.width/2, self.img.shape[0])
        y = np.linspace(self.y-self.height/2, self.y +
                        self.height/2, self.img.shape[1])
        xv, yv = np.meshgrid(x, y)
        X = xv.reshape(-1)
        Y = yv.reshape(-1)
        Z = X*0+0.001
        points = np.row_stack((X, Y, Z))
        return points.T

def nothing(x):
    pass
