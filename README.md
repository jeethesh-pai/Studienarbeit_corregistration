# Co-registration of Laser scanned images and camera images.

Here we can see a wrapper built around the official implementation of [Superpoint](https://github.com/magicleap/SuperPointPretrainedNetwork.git)

## Requirements
- pytorch
- openCV
- matplotlib
- numpy

## Usage
For running the script 
```
superpoint_wrapper.py
```
the following arguments are required 
```
--image - path to directory containing images - defaults to Image_test
--nms   - non maximal suppression distance which can be used to reduce the no. of detected points - defaults to 4
--gpu   - can be used if the prediction needs to be run on GPU instead of CPU - defaults to False
```

run the script using the command 
```
python superpoint_wrapper.py
```
after pasting the required images for evaluation. Here i have pasted a few images from HPatches dataset which can be downloaded [here](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz). Simply untar these files and paste the required files in the `Image_test` folder. 
