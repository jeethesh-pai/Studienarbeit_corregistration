import torch
import torchvision.transforms
from cv2 import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from photometric import ImgAugTransform
from utils import sample_homography, inv_warp_image_batch, warp_points, filter_points_batch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def points_to_2D(points: np.ndarray, H: int, W: int, img=None) -> np.ndarray:
    labels = np.zeros((H, W))
    if len(points.shape) <= 1:
        return labels
    if img is not None:
        img = img / 255.0
        image_copy = np.copy(img)
        image_copy[points[:, 0], points[:, 1]] = 1
        return image_copy
    else:
        if points.shape[0] > 0:
            labels[points[:, 0], points[:, 1]] = 1
    return labels


def show(img):
    img = torchvision.transforms.ToPILImage()(img)
    img.show()


class InstituteData(Dataset):
    def __init__(self, transform=None, task='train', **config: dict):
        super(InstituteData, self).__init__()
        self.caps_transform = transform
        self.superpoint_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        self.config = config
        if task == 'train':
            self.task = 'Train'
        elif task == 'test':
            self.task = 'Test'
        else:
            self.task = 'Validation'
        self.image_path = os.path.join(self.config['data']['root'], self.task)
        self.image_list = os.listdir(self.image_path)
        self.resize_shape = self.config['data']['preprocessing']['resize']
        self.photometric = self.config['data']['augmentation']['photometric']['enable']
        self.torch_photometric_transforms = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.25)])
        self.homographic = self.config['data']['augmentation']['homographic']['enable']
        self.warped_pair_params = self.config['data']['augmentation']['homographic']['homographies']['params']
        self.sample_homography = sample_homography

    def __getitem__(self, index: int) -> dict:
        sample = {}
        image = Image.open(os.path.join(self.image_path, self.image_list[index]))
        image = image.resize(self.resize_shape, Image.ANTIALIAS)
        image = transforms.ToTensor()(image)
        channel, height, width = image.shape
        if self.photometric:
            image = self.torch_photometric_transforms(image)
        if self.homographic:
            num_iter = self.config['data']['augmentation']['homographic']['num']
            # use inverse of homography as we have initial points which needs to be homographically augmented
            homographies = torch.stack([self.sample_homography(np.array([height, width]),
                                                               shift=0, **self.warped_pair_params)
                                        for i in range(num_iter)]).type(torch.float32)  # actual homography
            if torch.prod(torch.linalg.det(homographies)) == 0:
                while torch.prod(torch.linalg.det(homographies)) != 0:
                    homographies = torch.stack([self.sample_homography(np.array([image.shape[0], image.shape[1]]),
                                                                       shift=0, **self.warped_pair_params)
                                                for i in range(num_iter)]).type(torch.float32)
            sample['inv_homography'] = torch.linalg.inv(homographies)
            warped_image = torch.cat([image.unsqueeze(0)]*num_iter, dim=0)
            sample['homography'] = homographies
            sample['warped_image'] = inv_warp_image_batch(warped_image, mode='bilinear',
                                                          mat_homo_inv=sample['homography'])
            sample['warped_image_caps'] = self.caps_transform(sample['warped_image']).type(torch.float32)
            sample['warped_image_superpoint'] = (self.superpoint_transform(sample['warped_image'])).type(torch.float32)
        sample['caps_image'] = self.caps_transform(image).type(torch.float32)
        sample["superpoint_image"] = self.superpoint_transform(image).type(torch.float32)
        sample['name'] = self.image_list[index]
        return sample

    def __len__(self):
        return len(os.listdir(self.image_path))
