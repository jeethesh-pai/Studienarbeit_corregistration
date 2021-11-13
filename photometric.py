from imgaug import augmenters as iaa
import numpy as np
from cv2 import cv2
from numpy.random import randint


class ImgAugTransform:
    def __init__(self, **config: dict):
        if config['photometric']['enable']:
            params = config['photometric']['params']
            aug_all = []
            if params.get('random_brightness', False):
                change = params['random_brightness']['max_abs_change']
                aug = iaa.Add((-change, change))
                aug_all.append(aug)
            if params.get('random_contrast', False):
                change = params['random_contrast']['strength_range']
                aug = iaa.LinearContrast((change[0], change[1]))
                aug_all.append(aug)
            if params.get('additive_gaussian_noise', False):
                change = params['additive_gaussian_noise']['stddev_range']
                aug = iaa.AdditiveGaussianNoise(scale=(change[0], change[1]))
                aug_all.append(aug)
            if params.get('additive_speckle_noise', False):
                change = params['additive_speckle_noise']['prob_range']
                aug = iaa.ImpulseNoise(p=(change[0], change[1]))
                aug_all.append(aug)
            if params.get('motion_blur', False):
                change = params['motion_blur']['max_kernel_size']
                aug = iaa.Sometimes(0.05, iaa.MotionBlur(change))
                aug_all.append(aug)
            if params.get('GaussianBlur', False):
                change = params['GaussianBlur']['sigma']
                aug = iaa.GaussianBlur(sigma=change)
                aug_all.append(aug)

            self.aug = iaa.SomeOf(2, aug_all)
        else:
            self.aug = iaa.Sequential([iaa.Noop()])

    def __call__(self, img: np.ndarray):
        img = img.astype(np.uint8)
        img = self.aug.augment_image(img)
        return img