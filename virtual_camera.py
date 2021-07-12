import numpy as np


class VirtualCamera:
    def __init__(self, alpha, beta, gamma, Tx, Ty, Tz, focal, resolution) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        self.focal = focal
        self.resolution = np.array(resolution)
        self.center = np.array(resolution) / 2
        self.pixel_size_x = 3.45e-6
        self.pixel_size_y = 3.45e-6
        self.RT_camera_in_world = np.zeros((4, 4))
        self.RT_world_in_camera = np.zeros((4, 4))
        self.update()

    def look_at(self, point):
        origin = np.asarray([0, 0, 1])
        point = np.asarray(point)
        camera_xyz = np.asarray([self.Tx, self.Ty, self.Tz])
        vector = point - camera_xyz
        vector_norm = vector / np.linalg.norm(vector)
        axis = np.cross(vector_norm, origin)
        angle = np.arccos(np.dot(origin, vector_norm))

        R1 = self.axis_angle_to_rotation_matrix(axis, angle)

        T1 = np.array(
            [
                [1, 0, 0, -self.Tx],
                [0, 1, 0, -self.Ty],
                [0, 0, 1, -self.Tz],
                [0, 0, 0, 1],
            ]
        )
        self.RT_world_in_camera = np.matmul(R1, T1)
        print(self.RT_world_in_camera)

    def axis_angle_to_rotation_matrix(self, axis, angle):
        nx, ny, nz = axis
        ct = np.cos(angle)
        st = np.sin(angle)
        res = np.asarray(
            [
                [
                    ct + nx * nx * (1 - ct),
                    -nz * st + nx * ny * (1 - ct),
                    ny * st + nx * nz * (1 - ct),
                    0,
                ],
                [
                    nz * st + nx * ny * (1 - ct),
                    ct + ny * ny * (1 - ct),
                    -nx * st + ny * nz * (1 - ct),
                    0,
                ],
                [
                    -ny * st + nx * nz * (1 - ct),
                    nx * st + ny * nz * (1 - ct),
                    ct + nz * nz * (1 - ct),
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )
        return res

    def update(self):
        Rx = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(self.alpha), np.sin(self.alpha), 0],
                [0, -np.sin(self.alpha), np.cos(self.alpha), 0],
                [0, 0, 0, 1],
            ]
        )
        Ry = np.array(
            [
                [np.cos(self.beta), 0, -np.sin(self.beta), 0],
                [0, 1, 0, 0],
                [np.sin(self.beta), 0, np.cos(self.beta), 0],
                [0, 0, 0, 1],
            ]
        )
        Rz = np.array(
            [
                [np.cos(self.gamma), np.sin(self.gamma), 0, 0],
                [-np.sin(self.gamma), np.cos(self.gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        R1 = np.matmul(Rx, np.matmul(Ry, Rz))

        T1 = np.array(
            [
                [1, 0, 0, -self.Tx],
                [0, 1, 0, -self.Ty],
                [0, 0, 1, -self.Tz],
                [0, 0, 0, 1],
            ]
        )
        self.RT_world_in_camera = np.matmul(R1, T1)

    def project_world_to_camera(self, points):
        assert points.shape[1] == 3, "The n points' shape should be (n,3)"
        n = points.shape[0]
        points = points.T
        points = np.row_stack((points, np.ones((1, n))))

        points = self.RT_world_in_camera @ points
        points = points.T
        return points[:, 0:3]

    def project_camera_to_image(self, points):
        assert points.shape[1] == 3, "The n points' shape should be (n,3)"
        n = points.shape[0]
        points = points.T
        z = points[2, :]
        # print(z)
        trans = np.array([[self.focal, 0, 0], [0, self.focal, 0], [0, 0, 1]])
        points = trans @ points
        points = points / z
        points = points.T
        return points

    def project_image_to_pixel(self, points):
        assert points.shape[1] == 3, "The n points' shape should be (n,3)"
        n = points.shape[0]
        points = points.T
        trans = np.array(
            [
                [1 / self.pixel_size_x, 0, self.center[0]],
                [0, 1 / self.pixel_size_y, self.center[1]],
                [0, 0, 1],
            ]
        )
        points = trans @ points
        points = points.T
        return points

    def project_world_to_pixel(self, points):
        camera_points = self.project_world_to_camera(points)
        image_points = self.project_camera_to_image(camera_points)
        pixel_points = self.project_image_to_pixel(image_points)

        return pixel_points

    def crop_pixel(self, points, index_all):
        W = self.resolution[0]
        H = self.resolution[1]

        points = np.column_stack((points, index_all))

        index = np.where(points[:, 0] >= 0)
        index = np.squeeze(index)
        points = points[index, :]

        index = np.where(points[:, 0] <= W)
        index = np.squeeze(index)
        points = points[index, :]

        index = np.where(points[:, 1] >= 0)
        index = np.squeeze(index)
        points = points[index, :]

        index = np.where(points[:, 1] <= H)
        index = np.squeeze(index)
        points = points[index, :]

        points = points.astype(int)
        index = points[:, 3]
        points = points[:, 0:2]
        return points, index


if __name__ == "__main__":
    cam = VirtualCamera(0, np.pi / 2, 0, -1, 0, 0, 0.003, [1000, 1000])
    print(cam.RT_world_in_camera)
    p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    p = p.reshape(-1, 3)
    ans = cam.project_world_to_pixel(p)
    print(ans)
    # Ans should be [[7.45539187e+02 1.24170659e+03 1.00000000e+00]
    # [1.96595426e+02 1.95919429e+03 1.00000000e+00]]
    cam.look_at([0, 0, 0])
