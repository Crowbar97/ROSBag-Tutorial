import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import read_calib_file, load_image, load_point_cloud, project_velo_to_cam2, project_to_image


# draws lidar points on image
def draw_lidar_on_image(calib, img, point_cloud):
    # getting projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # getting image dims
    print('img shape: %s' % str(img.shape))
    img_height, img_width, _ = img.shape

    # apply projection
    print('point cloud: %s' % str(point_cloud.shape))
    points_on_img = project_to_image(point_cloud.transpose(), proj_velo2cam2)

    # finding lidar points to be within image FOV
    print('img: %s' % str(points_on_img.shape))
    inds = np.where((points_on_img[0, :] < img_width) & (points_on_img[0, :] >= 0) &
                    (points_on_img[1, :] < img_height) & (points_on_img[1, :] >= 0) &
                    (point_cloud[:, 0] > 0)
                    )[0]

    # filtering out pixels points
    imgfov_points = points_on_img[:, inds]
    print('filt: %s' % str(imgfov_points.transpose().shape))

    # retrieving depth from lidar
    imgfov_pc = point_cloud[inds, :]
    imgfov_pc = np.hstack((imgfov_pc, np.ones((imgfov_pc.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_points.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth) % 256, :]
        cv2.circle(img, (int(np.round(imgfov_points[0, i])),
                         int(np.round(imgfov_points[1, i]))),
                   2, color=tuple(color), thickness=-1)
    return img


def show(img):
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()


if __name__ == '__main__':
    # loading calibration
    calib = read_calib_file('./calib.txt')
    # loading image
    img = load_image('./left_images/00000000.png')
    # loading point cloud
    point_cloud = load_point_cloud('./point_clouds/00000000.bin')

    img = draw_lidar_on_image(calib, img, point_cloud)
    show(img)

