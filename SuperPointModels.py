import torch
import numpy as np
import torchvision.models
import matplotlib.pyplot as plt
from collections import namedtuple
from torchsummary import summary
from utils import nms_fast


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. From Daniel De Tone Implementation"""

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x: torch.Tensor) -> dict:
        """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return {'semi': semi, 'desc': desc}  # semi is the detector head and desc is the descriptor head

    def eval_mode(self, image: np.ndarray, conf_threshold: float, H: int, W: int, dist_thresh: float) -> tuple:
        with torch.no_grad():
            (_, semi), (_, desc) = self.forward(
                torch.from_numpy(image[np.newaxis, np.newaxis, :, :])).items()
            heatmap = semi_to_heatmap(semi)
            xs, ys = np.where(heatmap >= conf_threshold)
            if len(xs) == 0:
                return np.zeros((3, 0)), None, None
            pts = np.zeros((3, len(xs)))
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]
            pts, _ = nms_fast(pts, H, W, dist_thresh=dist_thresh)
            inds = np.argsort(-pts[2, :])  # sort by confidence
            bord = 4  # border to remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]
            #  -- process descriptor
            D = desc.shape[1]
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                samp_pts = torch.from_numpy(pts[:2, :].copy())
                samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
                samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
                samp_pts = samp_pts.transpose(0, 1).contiguous()
                samp_pts = samp_pts.view(1, 1, -1, 2)
                samp_pts = samp_pts.float()
                desc = torch.nn.functional.grid_sample(desc, samp_pts, align_corners=True)
                desc = desc.data.cpu().numpy().reshape(D, -1)
                desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :] + 1e-10
        return pts, desc, heatmap


@torch.no_grad()
def semi_to_heatmap(semi: torch.Tensor) -> np.ndarray:
    """
    This will work only if the tensor shape is [batch_size, channel, height, width]
    """
    assert len(semi.shape) == 4
    SoftMax = torch.nn.Softmax(dim=1)  # apply softmax on the channel dimension with 65
    soft_output = SoftMax(semi)
    soft_output = soft_output[:, :-1, :, :]
    pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=8)
    heatmap = pixel_shuffle(soft_output).squeeze()
    return heatmap


def detector_post_processing(semi: torch.Tensor, conf_threshold=0.015, NMS_dist=1, ret_heatmap=False,
                             limit_detection=600) -> np.ndarray:
    """
    :param semi - Output prediction from SuperpointNet with shape (65, Hc, Wc) - takes only one image at once
    :param conf_threshold - Detector confidence threshold to be applied
    :param NMS_dist - Correct distance used for Non maximal suppression
    :param ret_heatmap - returns heatmap with size (Hc x 8, Wc x 8)
    :param limit_detection - total no. of detection which needs to be considered utmost.
    """
    assert len(semi.shape) > 3
    with torch.no_grad():
        SoftMax = torch.nn.Softmax(dim=0)  # apply softmax on the channel dimension with 65
        soft_output = SoftMax(semi.squeeze())
        soft_output = soft_output[:-1, :, :]
        pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=8)
        heatmap = pixel_shuffle(soft_output).to('cpu').numpy().squeeze()
        if ret_heatmap:
            return heatmap
        xs, ys = np.where(heatmap >= conf_threshold)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        H, W = heatmap.shape
        pts, _ = nms_fast(pts, heatmap.shape[0], heatmap.shape[1], dist_thresh=NMS_dist)
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = 4  # we consider 4 pixels from all the boundaries as rejected
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        if pts.shape[1] > limit_detection:
            pts = pts[:, :limit_detection]
    return pts


class SuperPointNetBatchNorm(SuperPointNet):
    """ Pytorch definition of SuperPoint Network with added Batch Normalization. From Daniel De Tone Implementation"""

    def __init__(self):
        super(SuperPointNetBatchNorm, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Sequential(
            torch.nn.Conv2d(1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c1), torch.nn.ReLU(inplace=True))
        self.conv1b = torch.nn.Sequential(
            torch.nn.Conv2d(c1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c1), torch.nn.ReLU(inplace=True))
        self.conv2a = torch.nn.Sequential(
            torch.nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c2), torch.nn.ReLU(inplace=True))
        self.conv2b = torch.nn.Sequential(
            torch.nn.Conv2d(c2, c2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c2), torch.nn.ReLU(inplace=True))
        self.conv3a = torch.nn.Sequential(
            torch.nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c3), torch.nn.ReLU(inplace=True))
        self.conv3b = torch.nn.Sequential(
            torch.nn.Conv2d(c3, c3, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c3), torch.nn.ReLU(inplace=True))
        self.conv4a = torch.nn.Sequential(
            torch.nn.Conv2d(c3, c4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c4), torch.nn.ReLU(inplace=True))
        self.conv4b = torch.nn.Sequential(
            torch.nn.Conv2d(c4, c4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c4), torch.nn.ReLU(inplace=True))
        # Detector Head.
        self.convPa = torch.nn.Sequential(
            torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c5), torch.nn.ReLU(inplace=True))
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Sequential(
            torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=c5), torch.nn.ReLU(inplace=True))
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x: torch.Tensor) -> dict:
        """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
        # Shared Encoder.
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        # Detector Head.
        cPa = self.convPa(x)
        semi = self.convPb(cPa)

        # Descriptor Head.
        cDa = self.convDa(x)
        desc = self.convDb(cDa)
        dn = torch.linalg.norm(desc + 1e-10, dim=1)  # Compute the norm.
        desc_norm = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        # print('Descriptor_norm:', desc_norm.sum(dim=1), 'Shape: ', desc_norm.shape)
        return {'semi': semi, 'desc': desc_norm}  # semi is the detector head and desc is the descriptor head


class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues. This wraps model with dictionary outputs
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor):
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data


class ResNetSuperPoint(torch.nn.Module):
    """defines ResNet backbone for Superpoint uses first three layers of Resnet50"""
    def __init__(self):
        super(ResNetSuperPoint, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.maxpool1 = self.resnet.maxpool
        self.relu1 = self.resnet.relu

        # encoder
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        # Detector Head.
        self.convPa = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256), torch.nn.ReLU(inplace=True))
        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.convPa(x)
        semi = self.convPb(x)
        return {"semi": semi, "desc": semi}


