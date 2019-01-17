# Image Segmentation/ Pixel-Wise Classification

This model uses the [camvid motion-based segmentation and recognition dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) with a per-pixel segmentation of 700 images. The data was acquired from the point of view of a driving vehicle.

Here's an example of an original image. 

<img src="https://github.com/mlsmall/Image-Segmentation/blob/master/original.png" width="300" />

Here's an example of the segmentation label for the same image.  

<img src="https://github.com/mlsmall/Image-Segmentation/blob/master/segmented.png" width="300" />

Each pixel is color coded to represents a number that is associated to a class. The classes are as follow:

- 'Animal': 0
- 'Archway': 1
- 'Bicyclist': 2
- 'Bridge': 3
- 'Building': 4
- 'Car': 5
- 'CartLuggagePram': 6
- 'Child': 7
- 'Column_Pole': 8
- 'Fence': 9
- 'LaneMkgsDriv': 10
- 'LaneMkgsNonDriv': 11
- 'Misc_Text': 12
- 'MotorcycleScooter': 13
- 'OtherMoving': 14
- 'ParkingBlock': 15
- 'Pedestrian': 16
- 'Road': 17
- 'RoadShoulder': 18
- 'Sidewalk': 19
- 'SignSymbol': 20
- 'Sky': 21
- 'SUVPickupTruck': 22
- 'TrafficCone': 23
- 'TrafficLight': 24
- 'Train': 25
- 'Tree': 26
- 'Truck_Bus': 27
- 'Tunnel': 28
- 'VegetationMisc': 29
- 'Void': 30
- 'Wall': 31
 
## Model Acrhitecture

This model uses the [U-Net architecture](https://arxiv.org/abs/1505.04597), which has been traditionally used for biomedical image segmentation, and its a convulutional nueral network that contains 3 parts:

    1 : A contracting/ downsampling path
    2 : A bottleneck
    3 : An expanding/ upsampling path
    
<img src="https://github.com/mlsmall/Image-Segmentation/blob/master/unet.jpg" width="680" />

The contracting path is a typical convolutional network that consists convolutions each followed by a rectified linear unit (ReLU) and a max pooling operation. During the contraction, the spatial information is reduced while feature information is increased. The input is reduced to a linear feature representation. 

Then this linear feature representation is upsampled through the expanding path so that at the end the resulting image is of the same dimension as the input image. Hence these layers increase the resolution of the output

The linear compression of the input leads to a bottleneck that does not transmit all features. So the U-Net adds skip connections that allow feature characteristics to bypass the bottleneck section.

This particular U-Net uses a pre-trained [ResNet-34](https://arxiv.org/abs/1512.03385) network that was trained on [ImageNet](http://www.image-net.org/).

## Results

The result was an image segmentation model with an accuracy of 92.4% after going through only 10 epochs.

The images on the left represent the ground truth labels while the images on the right represent their predictions.


<img src="https://github.com/mlsmall/Image-Segmentation/blob/master/results.png" width="580" />
