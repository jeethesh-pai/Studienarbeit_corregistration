import cv2
import torch.cuda
from torchvision import transforms
from SuperPointPretrainedNetwork import demo_superpoint as superpoint
import os
import numpy as np
import matplotlib.pyplot as plt
from caps_implementation.CAPS.caps_model import CAPSModel, CAPSNet
import caps_implementation.config as config

# model = superpoint.SuperPointFrontend(weights_path='SuperPointPretrainedNetwork/superpoint_v1.pth',
#                                       nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=False)
model = superpoint.SuperPointFrontend(weights_path='SuperPointPretrainedNetwork/superpoint_v1.pth',
                                      nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=False, net="default")

# https://github.com/mmmfarrell/SuperPoint/blob/master/superpoint/match_features_demo.py
descriptor_matcher = superpoint.PointTracker(max_length=4, nn_thresh=0.7)


def extract_SIFT_keypoints(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(img_gray, None)
    return keypoint, descriptor


def extract_ORB_keypoints(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoint, descriptor = orb.detectAndCompute(img_gray, None)
    return keypoint, descriptor


def extract_BRIEF_Keypoints(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints = fast.detect(img_gray, None)
    keypoints, descriptor = brief.compute(img_gray, keypoints)
    return keypoints, descriptor


def extract_superpoint_keypoints(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.asarray(img_gray, dtype=np.float32) / 255.0
    keypoint, descriptor, heat_map = model.run(img_gray)
    keypoint = np.transpose(keypoint)
    keypoint = [cv2.KeyPoint(int(point[0]), int(point[1]), 1) for point in keypoint]
    return keypoint, descriptor


def extract_CAPSDescriptor(keypoint, img):
    args = config.get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.data_dir = "Dense_match"
    args.ckpt_path = "CAPS_grayscale_weights/150000.pth"
    # args.ckpt_path = "caps-pretrained.pth"
    descriptor_model = CAPSModel(args)
    img_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    img = torch.from_numpy(img).float().to(device) / 255.0
    img = torch.unsqueeze(img, dim=0)
    img = img.permute(0, 3, 1, 2)
    img = img_transform(img)
    keypoint = cv2.KeyPoint_convert(keypoint)
    keypoint = torch.Tensor(keypoint).to(device)
    keypoint = torch.unsqueeze(keypoint, dim=0).int()
    feat_c, feat_f = descriptor_model.extract_features(img, keypoint)
    descriptor = torch.cat((feat_c, feat_f), -1).squeeze(0).detach().cpu().numpy()
    return descriptor


def match_descriptors(kp1, descriptor1, kp2, descriptor2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search applicable only for SIFT
    bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]
    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1: list, matched_kp2: list) -> (np.ndarray, np.ndarray):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
    # Estimate the homography between the matches using RANSAC
    H, inlier = cv2.findHomography(matched_pts1[:, [1, 0]],
                                   matched_pts2[:, [1, 0]],
                                   cv2.RANSAC)
    inlier = inlier.flatten()
    return H, inlier


def offset_keypoint(keypoint: list, img1_shape: tuple) -> list:
    """
    This function offsets the keypoint of the second image for dense correspondence matching
    :param keypoint : list of cv2.KeyPoint objects of the second image whose x coordinates (width) needs to be offset
    :param img1_shape: shape of the first image, ideally the width of image is taken to offset the keypoint
    :return keypoint_new: offset keypoint which can be used for line drawing
    """
    point_convert = cv2.KeyPoint_convert(keypoint)
    point_convert[:, 0] = point_convert[:, 0] + img1_shape[1]
    keypoint_new = [cv2.KeyPoint(int(point_new[0]), int(point_new[1]), 1) for point_new in point_convert]
    return keypoint_new


def draw_matches_superpoint_caps(img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray):
    """outputs an image which shows correspondence between two images with the help of superpoint keypoint and
    caps descriptor
    :param img1 - image 1 - 3 channel or 1 channel (image will be converted to grayscale)
    :param img2 - image 2
    :returns combined_image - image showing correspondence matches
    :returns kp_image - returns image with keypoints marked in it
    """
    keypoint1, descriptor1 = extract_superpoint_keypoints(img1)
    keypoint2, descriptor2 = extract_superpoint_keypoints(img2)
    descriptor1 = extract_CAPSDescriptor(keypoint1, img1)
    descriptor2 = extract_CAPSDescriptor(keypoint2, img2)
    kp1_match, kp2_match, match = match_descriptors(keypoint1, descriptor1, keypoint2, descriptor2)
    new_keypoint = offset_keypoint(kp2_match, img1.shape)
    combined_keypoint = np.concatenate([kp1_match, new_keypoint], axis=0)
    combined_image = cv2.hconcat([img1, img2])
    kp_image = np.copy(combined_image)
    kp_image = cv2.drawKeypoints(kp_image, combined_keypoint, None, color=(0, 255, 0))
    h_mat, inlier_points = compute_homography(kp1_match, kp2_match)
    match = np.array(match)[inlier_points.astype(bool)].tolist()
    print("No. of matched points: ", len(match))
    combined_image = cv2.drawMatches(img1, keypoint1, img2, keypoint2, match, None,
                                     matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
    return combined_image, kp_image


def draw_matches_sift_caps(img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray):
    """outputs an image which shows correspondence between two images with the help of superpoint keypoint and
    caps descriptor
    :param img1 - image 1 - 3 channel or 1 channel (image will be converted to grayscale)
    :param img2 - image 2
    :returns combined_image - image showing correspondence matches
    :returns kp_image - returns image with keypoints marked in it
    """
    keypoint1, descriptor1 = extract_SIFT_keypoints(img1)
    keypoint2, descriptor2 = extract_SIFT_keypoints(img2)
    descriptor1 = extract_CAPSDescriptor(keypoint1, img1)
    descriptor2 = extract_CAPSDescriptor(keypoint2, img2)
    kp1_match, kp2_match, match = match_descriptors(keypoint1, descriptor1, keypoint2, descriptor2)
    new_keypoint = offset_keypoint(kp2_match, img1.shape)
    combined_keypoint = np.concatenate([kp1_match, new_keypoint], axis=0)
    combined_image = cv2.hconcat([img1, img2])
    kp_image = np.copy(combined_image)
    kp_image = cv2.drawKeypoints(kp_image, combined_keypoint, None, color=(0, 255, 0))
    h_mat, inlier_points = compute_homography(kp1_match, kp2_match)
    match = np.array(match)[inlier_points.astype(bool)].tolist()
    combined_image = cv2.drawMatches(img1, keypoint1, img2, keypoint2, match, None,
                                     matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
    return combined_image, kp_image


def draw_matches_superpoint(img1: np.ndarray, img2: np.ndarray, nn_thresh: float) -> \
        (np.ndarray, np.ndarray):
    """outputs an image which shows correspondence between two images with the help of superpoint descriptor and
    keypoint detector
    :param img1 - image 1 - 3 channel or 1 channel (image will be converted to grayscale)
    :param img2 - image 2
    :param nn_thresh - threshold use to reduce no. of outlier matched, ideally the nearest neighbour threshold
    :returns combined_image - image showing correspondence matches
    :returns kp_image - returns image with keypoints marked in it
    """
    keypoint1, descriptor1 = extract_superpoint_keypoints(img1)
    keypoint2, descriptor2 = extract_superpoint_keypoints(img2)
    match = descriptor_matcher.nn_match_two_way(descriptor1, descriptor2, nn_thresh=nn_thresh)
    match_desc1_idx = np.array(match[0, :], dtype=int)  # descriptor 1 matches
    match_desc2_idx = np.array(match[1, :], dtype=int)  # descriptor 2 matches
    matched_keypoint1 = [keypoint1[idx] for idx in match_desc1_idx]
    matched_keypoint2 = [keypoint2[idx] for idx in match_desc2_idx]
    new_keypoint = offset_keypoint(keypoint2, img1.shape)
    combined_keypoint = np.concatenate([keypoint1, new_keypoint], axis=0)
    combined_image = cv2.hconcat([img1, img2])
    kp_image = np.copy(combined_image)
    kp_image = cv2.drawKeypoints(kp_image, combined_keypoint, None, color=(0, 255, 0))
    match_point1 = cv2.KeyPoint_convert(matched_keypoint1)
    match_point2 = cv2.KeyPoint_convert(matched_keypoint2)
    H, inlier = cv2.findHomography(match_point1[:, [1, 0]], match_point2[:, [1, 0]], cv2.RANSAC)
    inlier = inlier.flatten()
    inlier_index = np.nonzero(inlier)
    match_point1 = np.squeeze(match_point1[inlier_index, :])
    match_point2 = np.squeeze(match_point2[inlier_index, :])
    match_point1 = [cv2.KeyPoint(int(point[0]), int(point[1]), 1) for point in match_point1]
    match_point2 = [cv2.KeyPoint(int(point[0]), int(point[1]), 1) for point in match_point2]
    match_point2 = offset_keypoint(match_point2, img1.shape)
    combined_keypoint = np.concatenate([match_point1, match_point2], axis=0)
    combined_image = cv2.drawKeypoints(combined_image, combined_keypoint, None, color=(0, 255, 0))
    match_point2 = cv2.KeyPoint_convert(match_point2)
    match_point1 = cv2.KeyPoint_convert(match_point1)
    for i in range(len(match_point1)):
        point1_i = (int(match_point1[i][0]), int(match_point1[i][1]))
        point2_i = (int(match_point2[i][0]), int(match_point2[i][1]))
        combined_image = cv2.line(combined_image, point1_i, point2_i, color=(0, 255, 0),
                                  thickness=2)
    return combined_image, kp_image


def draw_matches_orb(img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray):
    points1, desc1 = extract_ORB_keypoints(img1)
    points2, desc2 = extract_ORB_keypoints(img2)
    kp1_match, kp2_match, match = match_descriptors(points1, desc1, points2, desc2)
    new_keypoint = offset_keypoint(kp2_match, img1.shape)
    combined_keypoint = np.concatenate([kp1_match, new_keypoint], axis=0)
    combined_image = cv2.hconcat([img1, img2])
    kp_image = cv2.drawKeypoints(combined_image, combined_keypoint, None, color=(0, 255, 0))
    h_mat, inlier_points = compute_homography(kp1_match, kp2_match)
    match = np.array(match)[inlier_points.astype(bool)].tolist()
    print("No. of matched points: ", len(match))
    matched_img = cv2.drawMatches(img1, points1, img2, points2, match, None,
                                  matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
    return matched_img, kp_image


def draw_matches_surf(img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray):
    points1, desc1 = extract_BRIEF_Keypoints(img1)
    points2, desc2 = extract_BRIEF_Keypoints(img2)
    kp1_match, kp2_match, match = match_descriptors(points1, desc1, points2, desc2)
    new_keypoint = offset_keypoint(kp2_match, img1.shape)
    combined_keypoint = np.concatenate([kp1_match, new_keypoint], axis=0)
    combined_image = cv2.hconcat([img1, img2])
    kp_image = cv2.drawKeypoints(combined_image, combined_keypoint, None, color=(0, 255, 0))
    h_mat, inlier_points = compute_homography(kp1_match, kp2_match)
    match = np.array(match)[inlier_points.astype(bool)].tolist()
    print("No. of matched points: ", len(match))
    matched_img = cv2.drawMatches(img1, points1, img2, points2, match, None,
                                  matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
    return matched_img, kp_image


def draw_matches_sift(img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray):
    points1, desc1 = extract_SIFT_keypoints(img1)
    points2, desc2 = extract_SIFT_keypoints(img2)
    kp1_match, kp2_match, match = match_descriptors(points1, desc1, points2, desc2)
    new_keypoint = offset_keypoint(kp2_match, img1.shape)
    combined_keypoint = np.concatenate([kp1_match, new_keypoint], axis=0)
    combined_image = cv2.hconcat([img1, img2])
    kp_image = cv2.drawKeypoints(combined_image, combined_keypoint, None, color=(0, 255, 0))
    h_mat, inlier_points = compute_homography(kp1_match, kp2_match)
    match = np.array(match)[inlier_points.astype(bool)].tolist()
    print("No. of matched points: ", len(match))
    matched_img = cv2.drawMatches(img1, points1, img2, points2, match, None,
                                  matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
    return matched_img, kp_image


# # (width, height) = (768, 512)
# (width, height) = (640, 480)
# folder = 'Dense_match/'
# file_name = os.listdir(folder)
# image1 = cv2.imread(folder + file_name[0])
# image1 = cv2.resize(image1, (width, height), interpolation=cv2.INTER_AREA)
# image2 = cv2.imread(folder + file_name[1])
# image2 = cv2.resize(image2, (width, height), interpolation=cv2.INTER_AREA)
# matched_image, keypoint_image = draw_matches_superpoint_caps(image1, image2)
# # matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
# # keypoint_image = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB)
# # matched_image, kp_image = draw_matches_orb(image1, image2)
# cv2.imwrite("Results/library_superpoint_author_0_caps_untrained.png", matched_image)
# matched_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
# plt.imshow(matched_image)
# plt.show()
