import torchvision.transforms
import numpy as np
from DataLoader import InstituteData
import torch
import yaml
import argparse
from SuperPointModels import SuperPointNetBatchNorm, SuperPointNet
from caps_implementation.CAPS.caps_model import CAPSModel
import caps_implementation.config as config_caps
from torchvision import transforms
import cv2
from SuperPointModels import detector_post_processing
from dense_correspondense_matcher import match_descriptors, compute_homography
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="This scripts helps to evaluate descriptor using different metrics")
parser.add_argument('--config', help='Path to config file', default="descriptor_evaluation_config.yaml")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config = args.config

# load the config file
with open(config) as path:
    config = yaml.full_load(path)


def show(img):
    img = torchvision.transforms.ToPILImage()(img)
    img.show()


def extract_superpoint_keypoints(semi: torch.Tensor):
    points = detector_post_processing(semi, conf_threshold=0.015, NMS_dist=4, limit_detection=1000)
    keypoint = np.transpose(points)
    keypoint = [cv2.KeyPoint(int(point[0]), int(point[1]), 1) for point in keypoint]
    return keypoint


def convertToTorchKeypoints(keypoint_original, device):
    keypoint = np.copy(keypoint_original)
    keypoint = cv2.KeyPoint_convert(keypoint)
    keypoint = torch.Tensor(keypoint).to(device)
    keypoint = torch.unsqueeze(keypoint, dim=0).int()
    return keypoint


def warp_points(points: np.ndarray, homography: np.ndarray):
    homogenous_points = np.concatenate([points, np.ones(shape=(points.shape[0], 1))], axis=1)
    warped_points = homography @ homogenous_points.transpose()
    warped_points = warped_points.transpose()
    return warped_points[:, :2] / (warped_points[:, 2:] + 1e-6)  # small number to eliminate division by zero


def estimate_homography(homo_mat_true: np.ndarray, homo_mat_pred: np.ndarray, img_shape: tuple):
    points = np.asarray([[0, 0], [img_shape[0], 0], [img_shape[0], img_shape[1] - 1], [0, img_shape[1]]])
    warped_true = warp_points(points, homo_mat_true)
    warped_pred = warp_points(points, homo_mat_pred)
    mean_dist = np.linalg.norm(warped_true - warped_pred, axis=1)
    pass


batch_size = config['model']['batch_size']
# Load the both the models for detector and descriptor
detector_weights = torch.load(config['detector_weights'], map_location=device)
SuperPointModel = SuperPointNet()
SuperPointModel.load_state_dict(detector_weights)
SuperPointModel.train(mode=False)
args_caps = config_caps.get_args()
shape = config['data']['preprocessing']['resize']  # width, height
args_caps.ckpt_path = config['CAPS_weights']
CAPSModel = CAPSModel(args_caps)
caps_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
# caps_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
#                                     ])
ImageDataGenerator = InstituteData(task='val', transform=caps_transform, **config)
data_loader = torch.utils.data.DataLoader(ImageDataGenerator, batch_size=batch_size, shuffle=False)
for sample in data_loader:
    sample['caps_image'] = sample['caps_image'].to(device)
    sample['superpoint_image'] = sample['superpoint_image'].to(device)
    sample['warped_image_superpoint'] = sample['warped_image_superpoint'].to(device)
    sample['warped_image_caps'] = sample['warped_image_caps'].to(device)
    out = SuperPointModel(sample['superpoint_image'])
    pts = extract_superpoint_keypoints(out['semi'])
    pts_caps = convertToTorchKeypoints(pts, device)
    feat_c, feat_f = CAPSModel.extract_features(sample['caps_image'], pts_caps)
    descriptor = torch.cat((feat_c, feat_f), -1).squeeze(0).detach().cpu().numpy()
    out_warped = SuperPointModel(sample['warped_image_superpoint'])
    pts_warped = extract_superpoint_keypoints(out_warped['semi'])
    pts_warped_caps = convertToTorchKeypoints(pts_warped, device)
    feat_c_warped, feat_f_warped = CAPSModel.extract_features(sample['warped_image_caps'], pts_warped_caps)
    descriptor_warped = torch.cat((feat_c_warped, feat_f_warped), -1).squeeze(0).detach().cpu().numpy()
    match_kp1, match_kp2, matches = match_descriptors(pts, descriptor, pts_warped, descriptor_warped)
    h_mat, inlier_points = compute_homography(match_kp1, match_kp2)
    match = np.array(matches)[inlier_points.astype(bool)].tolist()
    estimate_homography(sample['homography'].squeeze().numpy(), h_mat, shape)
    img1 = (sample['superpoint_image'].numpy().squeeze() * 255).astype(np.uint8)
    # cv2.imwrite('Dense_match/img1.jpg', img1)
    img2 = (sample['warped_image_superpoint'].numpy().squeeze() * 255).astype(np.uint8)
    # cv2.imwrite('Dense_match/img2.jpg', img2)
    img1_pred_warp = cv2.warpPerspective(img1, h_mat, dsize=shape, flags=cv2.INTER_LINEAR)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1, cmap='gray')
    axes[1].imshow(img1_pred_warp, cmap='gray')
    plt.show()
    combined_image = cv2.drawMatches(img1, pts, img2, pts_warped, match, None, matchColor=(0, 255, 0),
                                     singlePointColor=(0, 0, 255))
    plt.imshow(combined_image, cmap='gray')
    plt.show()
