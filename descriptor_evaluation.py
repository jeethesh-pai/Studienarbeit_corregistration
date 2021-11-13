from DataLoader import InstituteData
import torch
import yaml
import argparse
from SuperPointModels import SuperPointNetBatchNorm, SuperPointNet
from caps_implementation.CAPS.caps_model import CAPSModel
import caps_implementation.config as config_caps
from torchvision import transforms

parser = argparse.ArgumentParser(description="This scripts helps to evaluate descriptor using different metrics")
parser.add_argument('--config', help='Path to config file', default="descriptor_evaluation_config.yaml")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config = args.config

# load the config file
with open(config) as path:
    config = yaml.load(path)

batch_size = config['model']['batch_size']
# Load the both the models for detector and descriptor
if config['detector_weights'] is not None:
    detector_weights = torch.load(config['detector_weights'], map_location=device)
    SuperPointModel = SuperPointNet()
    SuperPointModel.load_state_dict(detector_weights)
    SuperPointModel.train(mode=False)
args_caps = config_caps.get_args()
args_caps.ckpt_path = "CAPS_grayscale_weights/150000.pth"
CAPSModel = CAPSModel(args_caps)
caps_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
ImageDataGenerator = InstituteData(task='val', transform=caps_transform, **config)
data_loader = torch.utils.data.DataLoader(ImageDataGenerator, batch_size=batch_size, shuffle=False)
for sample in data_loader:
    image = sample["warped_image"]
    print(sample)
