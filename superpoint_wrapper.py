import cv2
import matplotlib.pyplot as plt
import os
from SuperPointPretrainedNetwork import demo_superpoint as superpoint
import argparse
import numpy as np

"""
This script marks keypoints on the given set of images with help of Superpoint algorithm. Please look in to requirements
for details regarding installation. For this script images inside Image_test folder will be taken for analysis
Please make sure not to paste more than 10 photos for better visualization of marked keypoints."""

parser = argparse.ArgumentParser(description='Superpoint detector -- works only for maximum of 10 images at a time')
parser.add_argument('--image', default='Image_test', help='Image directory for look up')
parser.add_argument('--height', default=120, type=int, help='height of the image if resize is needed')
parser.add_argument('--width', default=160, type=int, help='width of the image if resize is needed')
parser.add_argument('--nms', default=4, type=int, help='a parameter that reduces the no. of points detected by '
                                                       'imposing stricter conditions')
parser.add_argument('--gpu', help='toggles the program to run on gpu', default=False, type=bool)
args = parser.parse_args()
print(args)
list_dir = os.listdir(args.image)
model = superpoint.SuperPointFrontend(weights_path='SuperPointPretrainedNetwork/superpoint_v1.pth',
                                      nms_dist=args.nms, conf_thresh=0.015, nn_thresh=0.7, cuda=args.gpu)

if len(list_dir) == 0:
    raise IOError('Input directory does not exist or there are no images in it...')
elif len(list_dir) % 2 != 0:
    rows = 1
    cols = len(list_dir)
    fig, axes = plt.subplots(rows, cols)
    for j, file in enumerate(list_dir):
        file_name = args.image + '/' + file
        image = cv2.imread(file_name)
        image = cv2.resize(image, (args.width, args.height))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = np.asarray(image_gray, dtype=np.float32) / 255.0
        points, descriptor, heat_map = model.run(image_gray)
        points, indices = model.nms_fast(points, image.shape[0], image.shape[1], dist_thresh=args.nms)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(points.shape[1]):
            image = cv2.circle(image, (int(points[0, i]), int(points[1, i])), 2, (255, 0, 255), -1)
        axes[j].imshow(image)
else:
    rows = 2
    cols = len(list_dir) // rows
    fig, axes = plt.subplots(rows, cols)
    counts = 0
    for j, images in enumerate(list_dir):
        file_name = args.image + '/' + images
        image = cv2.imread(file_name)
        image = cv2.resize(image, (args.width, args.height))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = np.asarray(image_gray, dtype=np.float32) / 255.0
        points, descriptor, heat_map = model.run(image_gray)
        points, indices = model.nms_fast(points, image.shape[0], image.shape[1], dist_thresh=args.nms)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(points.shape[1]):
            image = cv2.circle(image, (int(points[0, i]), int(points[1, i])), 2, (255, 0, 255), -1)
        if j == cols:
            counts = counts + 1
        axes[counts, j % cols].imshow(image)
plt.show()