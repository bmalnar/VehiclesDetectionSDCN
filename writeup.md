# **Vehicles Detection Project**
## Writeup / README

This is a summary of the work done to develop a processing pipeline for the vehicles detection project for the Udacity Self-Driving Car Nanodegree. The github repositroy of the project can be found [here](https://github.com/bmalnar/VehiclesDetectionSDCN)

More information can be found in the jupyter notebook provided, and here we only provide the high level overview of the pipeline. 

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

