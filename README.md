# TrafficSignRecognition
A Deep Neural Network to do traffic sign recognition
* A lot has happened since MobilEye developed the first commercially deployed traffic sign recognition system in collaboration with Continental AG for the BMW-7 series vehicles. Quite a few vehicles have used this technology since. 
* Establishing a reliable Traffic Sign Classification mechanism is a major step in our journey towards building semi-autonomous/autonomous driving systems.
* This post intends to explain an approach to solve the problem of traffic sign classification and I intend to show how easy it is, to build, train and deploy a deep learning network for traffic sign classification.

# Highlights of this approach
* The traffic sign dataset that we will be working on is GTSRB — German Traffic Signs. 
* The approach used is deep learning.
* The type of neural network used is a Convolutional Neural Network (CNN) paired with a linear classifier.
* Python is the language used to program this.
* The complete source code can be found here
# Getting Started 
The pickled dataset containing 32x32x3 color images split across labelled training, test and validation sets
https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip

# Clone the reposiory
https://github.com/vamsiramakrishnan/TrafficSignRecognition.git

# Getting around the repository
* Initially, all the code resided in the file TrafficSignClassifier.ipynb
* To clean up the code and make it re-usable, part of the code was exported to a python file called PreProcessing.py which is used by the final version of the file to call functions. 
* There are deviations in the way the project has been designed and the questions answered. 

# Some highlights 
* 98.5% accuracy over the test set 
* Learning rate annealing, Dropout increment, batch size increase as accuracy increases
* Greedy best save implemented on validation accuracy being the criteria
* Dataset size varied and finally 

# Steps followed in 
# Download and Visualize
<p align="center">
  <img src="DataVisualization.png" width="350"/>
</p>
* Use pandas and matplotlib along with the SignNames.csv to visualize data

# Pre-Process Data
<p align="center">
  <img src="DataPreProcessingSteps.png" width="500"/>
</p>
* Pre-processing techniques include 
  1. Centering around mean globally
  2. Locally centering the image around the mean
  3. Normalizing using Standard Deviation
  4. Use Histogram Equalization

# Augment Data
* Use batch iteration to process data.
* Augment data based on a fixed sample size per class . Which means classes with lesser samples will be upscaled to arrive at the target sample number. 
* Some methods to augment data would be jittering using **projective transform** , **scaling**, **zooming** , **brightness A

# Model Architecture 
The model architecture is very similar to most classification deep learning nets. 
<p align="center">
  <img src="ModelArch.png" width="1000"/>
</p>
* Spatial Transformers have been deployed to achieve invariance in scale
* CNN for feature extraction
* Linear Classifier with softmax activation for the classification task

**Localization Modules ** -> **Spatial Transformer Module** -> **CNN** -> ** Linear Classifier**
**VGG1** -> **VGG2** -> **VGG3** -> **VGG4** -> **CONCAT - VGG1_VGG2_VGG3_VGG4** -> **FC1** -> **FC2** -> **Logits**

* VGG Net Blocks that perform convolutions 
* Each VGGNet block has two 2D Convolutions and a subsampling layer ( Max Pooling )
* We use 4 layers of VGG Net to increase depth from 32 to 128 
* We use Multi-scale convolutions and concatenate them with the VGG blocks using subsampling before we connect them to the fully connected layer

