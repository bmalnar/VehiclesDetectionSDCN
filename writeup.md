# **Vehicles Detection Project**
## Writeup / README

This is a summary of the work done to develop a processing pipeline for the vehicles detection project for the Udacity Self-Driving Car Nanodegree. The github repositroy of the project can be found [here](https://github.com/bmalnar/VehiclesDetectionSDCN)

More information can be found in the jupyter notebook provided, and here we only provide the high level overview of the pipeline. 

### Data loading and exploration

We load the data and visualize it in at the top of the notebook. There are 'car' and 'noncar' data provided by Udacity. 

We can see that the car images are mostly showing cars captured from behind, or slightly from the side. This is what we need for training in this case, because we typically see other vehicles on the highway from behind/side in the project video. For general case, we need more variety of angles. An example is shown below:

<img src="output_images/car_image.png" width="240" alt="Car Image" />

If we look at the images in the noncar distribution, we see images with road side and empty roads, which is typically what we see looking forward from the car. A typical example is shown below:

<img src="output_images/non_car_image.png" width="240" alt="Noncar Image" />

### Fucntions for feature extraction

We can extract three different types of features for this project:

* HOG features (function get_hog_features): histograms of oriented gradients computed separately for each channel in the image. If we have a color image(3 channels) we can decide whether to use all 3 channels or any combination of them. From the experiments performed for this project, HOG features seem to be more robust for the purpose of car detection
* Spatial features (function bin_spatial): here we attempt to recognize the car based on the template matching
* Color features (function color_hist): attempt to recognize the car based on the histogram of colors in the image

The importance of defining the feature vector is to get the pipeline that detects the cars robustly, but also the size of the vector should be looked at to ensure relatively fast detections (the lower the vector size, the faster the detection process). 

### Illustration of HOG 

Below we show how the HOG feature looks like for a car and a noncar image. These features can take a lot different look for different images, especially in the case of noncar images. For images of cars from behind, the HOG feature vector should be relatively robust. 

Car image:

<img src="output_images/car_image.png" width="240" alt="Car Image" />

Car HOG:

<img src="output_images/car_hog.png" width="240" alt="Car HOG" />

Noncar image:

<img src="output_images/non_car_image.png" width="240" alt="Noncar Image" />

Noncar HOG:

<img src="output_images/noncar_hog.png" width="240" alt="Noncar HOG" />

### Configuration of the feature extraction pipeline

We can run different experiments to pick up the best set of params for our applications. The following experiments were performed:

* Use a different combination of features. Using HOG should always be turned on because that brings the best results. For spatial and color histogram features, it was not clear that they contribute to the quality of the results. For that reason, they are turned off, which makes the size of the feature vector smaller. 
* Color space is eventually set to YCrCb. This has shown to be a good color space for HOG, and we are using all 3 channels and corresponding HOG features. 
* HOG params such as number of orientations, pixels per cell and cells per block are investigated, and the values listed below are shown to perform well. 

```
# Params for feature extraction
# Color space can be RGB, HSV, LUV, HLS, YUV, YCrCb
color_space = 'YCrCb' 
# Number of HOG orientations
orient = 11 
# Number of HOG pixels per cell
pix_per_cell = 16
# Number of HOG cells per block
cell_per_block = 2 
# HOG channel can be 0, 1, 2, or "ALL"
hog_channel = 'ALL' 
# Spatial binning dimensions
spatial_size = (16, 16) 
# Number of histogram bins
hist_bins = 32    
# Whether or not to use spatial features
spatial_feat = False 
# Whether or not to use color histogram features
hist_feat = False 
# Whether or not to use HOG features
hog_feat = True 
```

### Processing pipeline

The processing pipeline performs 4 major steps:

* Implement the sliding window search for the cars in the image. This is done by practically dividing the input image into windows of different sizes/scales, and looking at each window for a car. If the car is found, the window is remembered, or otherwise it is discarded. We use a linear SVM classifier trained on the input data provided by Udacity. 
* Based on the windows detected, we create a heatmap of the image, where there is more "heat" in the areas where detections are achieved with larger confidence. 
* Based on the heatmap, we generate labelled blobs where the cars ar epresumably detected
* Finally, based on these labels we draw rectangles on top of the input image, so that we can visualize the quality of the detection. 

## Training process

Udacity provided the training/test datasets extracted from the KITTI dataset. There are 'car' and 'noncar' images, which are processed and used in the training/testing phase to obtained the trained classifier. 

We investigate 3 different feature types for the pipeline (more information can be found in the notebook):

* HOG (histogram of oriented gradients)
* Color histogram
* Spatial/template feature

Eventually we decided to use only the HOG features, because it was not conclusive that the other two feature types contributed to the quality of the results. 

### Discussion 

The resulting video shows that the pipeline performs relatively well for the purpose of detecting vehicles on the road, but the major issue is that there are often false positives especially on the left hand side of the video. These could be further eliminated by increasing the quality of the processing pipeline (for example if we could obtain more training data), or by further post processing steps. The latter is probably easier, especially since these false positives only appear for 1-2 frames, and can be easily detected and discarded. 

Comparing the results to some of the other experiments I did based on neural networks (such as SSD detector architectures or scene segmentation architectures such as ICNet or PSPnet), using HOG and SVM is probably inferior in terms of the quality of results. However, the nice thing about this experiment is that we get increasing knowledge of how classical computer vision approaches work, to better estimate what is possible and what is to be epxected. Not everything in the car needs to be based on deep learning, and having a variaty of tools to attack the problem can only be beenfitial. 

