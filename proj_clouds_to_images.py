from os import path, listdir
import argparse

import cv2
import numpy as np

import rospy
from sensor_msgs.point_cloud2 import create_cloud, PointField, PointCloud2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header

from lid2cam import draw_lidar_on_image
from utils import read_calib_file, load_image, load_point_cloud


def to_msg(points):
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    header = Header(stamp=rospy.Time(), frame_id='points')
    msg = create_cloud(header, fields, points)
    return msg


def main(args):
    rospy.init_node('projection')
    rate = rospy.Rate(0.2)

    # CLOUD PUB
    cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)

    # IMAGE PUB
    cv_bridge = CvBridge()
    image_pub = rospy.Publisher('/image', Image, queue_size=10)

    # loading calibration
    calib = read_calib_file(args.calib_file_path)

    # PROJ PUB
    proj_pub = rospy.Publisher('/proj', Image, queue_size=10)

    image_file_names = sorted(listdir(args.image_dir_path))
    # print(image_file_names)

    for image_file_name in image_file_names:
        idx = image_file_name[:-len('.png')]
        print('IDX: %s' % idx)

        # CLOUD
        pc_file_path = path.join(args.pc_dir_path, '%s.bin' % idx)
        points = load_point_cloud(pc_file_path)
        cloud_pub.publish(to_msg(points))

        # IMAGE
        image_file_path = path.join(args.image_dir_path, image_file_name)
        image = load_image(image_file_path)
        image_pub.publish(cv_bridge.cv2_to_imgmsg(image))

        # PROJ
        proj = draw_lidar_on_image(calib, image, points)
        proj_pub.publish(cv_bridge.cv2_to_imgmsg(proj))

        rate.sleep()

    rospy.spin()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc', '--pc_dir_path', type=str, required=True,
                        help='Path to directory with point cloud files')
    parser.add_argument('-i', '--image_dir_path', type=str, required=True,
                        help='Path to directory with corresponding images')
    parser.add_argument('-c', '--calib_file_path', type=str, required=True,
                        help='Path to camera calibration file')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    
    main(args)