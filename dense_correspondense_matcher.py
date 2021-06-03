import cv2
from SuperPointPretrainedNetwork import demo_superpoint as superpoint
import os
import numpy as np
import matplotlib.pyplot as plt

model = superpoint.SuperPointFrontend(weights_path='SuperPointPretrainedNetwork/superpoint_v1.pth',
                                      nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=False)

# https://github.com/mmmfarrell/SuperPoint/blob/master/superpoint/match_features_demo.py
descriptor_matcher = superpoint.PointTracker(max_length=4, nn_thresh=0.7)


def extract_SIFT_keypoints(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(img_gray, None)
    return keypoint, descriptor


def extract_superpoint_keypoints(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.asarray(img_gray, dtype=np.float32) / 255.0
    keypoint, descriptor, heat_map = model.run(img_gray)
    keypoint = np.transpose(keypoint)
    keypoint = [cv2.KeyPoint(int(point[0]), int(point[1]), 1) for point in keypoint]
    return keypoint, descriptor


def match_descriptors(kp1, descriptor1, kp2, descriptor2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inlier = cv2.findHomography(matched_pts1[:, [1, 0]],
                                   matched_pts2[:, [1, 0]],
                                   cv2.RANSAC)
    inlier = inlier.flatten()
    return H, inlier


def offset_keypoint(keypoint, img1_shape):
    point_convert = cv2.KeyPoint_convert(keypoint)
    point_convert[:, 0] = point_convert[:, 0] + img1_shape[1]
    keypoint_new = [cv2.KeyPoint(int(point_new[0]), int(point_new[1]), 1) for point_new in point_convert]
    return keypoint_new


def draw_matches(img1, keypoint1, img2, keypoint2, descriptor1, descriptor2, nn_thresh):
    match = descriptor_matcher.nn_match_two_way(descriptor1, descriptor2, nn_thresh=nn_thresh)
    match_desc1_idx = np.array(match[0, :], dtype=int)  # descriptor 1 matches
    match_desc2_idx = np.array(match[1, :], dtype=int)  # descriptor 2 matches
    matched_keypoint1 = [keypoint1[idx] for idx in match_desc1_idx]
    matched_keypoint2 = [keypoint2[idx] for idx in match_desc2_idx]
    new_keypoint = offset_keypoint(matched_keypoint2, img1.shape)
    combined_keypoint = np.concatenate([matched_keypoint1, new_keypoint], axis=0)
    combined_image = cv2.hconcat([img1, img2])
    combined_image = cv2.drawKeypoints(combined_image, combined_keypoint, None, color=(0, 255, 0))
    match_point1 = cv2.KeyPoint_convert(matched_keypoint1)
    # match_point_3 = cv2.KeyPoint_convert(matched_keypoint2)
    # H, inlier = cv2.findHomography(match_point1[:, [1, 0]], match_point_3[:, [1, 0]], cv2.RANSAC)
    # inlier = inlier.flatten()
    # match_point1 = np.array(match_point1)[inlier.astype(bool)].tolist()
    # match_point2 = np.array(match_point1)[inlier.astype(bool)].tolist()
    # match_point2 = offset_keypoint(match_point2, img1.shape)
    match_point2 = cv2.KeyPoint_convert(new_keypoint)
    for i in range(len(match_point1)):
        point1_i = (int(match_point1[i][0]), int(match_point1[i][1]))
        point2_i = (int(match_point2[i][0]), int(match_point2[i][1]))
        combined_image = cv2.line(combined_image, point1_i, point2_i, color=(0, 255, 0),
                                  thickness=2)
    return combined_image


(width, height) = (768, 512)
folder = 'Dense_match/'
file_name = os.listdir(folder)
image1 = cv2.imread(folder + file_name[0])
image1 = cv2.resize(image1, (width, height))
points1, desc1 = extract_superpoint_keypoints(image1)
# points1, desc1 = extract_SIFT_keypoints(image1)
image2 = cv2.imread(folder + file_name[1])
image2 = cv2.resize(image2, (width, height))
points2, desc2 = extract_superpoint_keypoints(image2)
# points2, desc2 = extract_SIFT_keypoints(image1)
# kp1_match, kp2_match, match = match_descriptors(points1, desc1, points2, desc2)
# h_mat, inlier_points = compute_homography(kp1_match, kp2_match)
# match = np.array(match)[inlier_points.astype(bool)].tolist()
# matched_img = cv2.drawMatches(image1, points1, image2, points2, match, None,
#                               matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
points2_offset = offset_keypoint(points2, image1.shape)
keypoint_image = cv2.hconcat([image1, image2])
total_keypoints = np.concatenate([points1, points2_offset], axis=0)
keypoint_image = cv2.drawKeypoints(keypoint_image, total_keypoints, None, color=(0, 255, 0))
plt.imshow(keypoint_image)
plt.show()
matched_img = draw_matches(image1, points1, image2, points2, desc1, desc2, 0.8)
plt.imshow(matched_img)
plt.show()
