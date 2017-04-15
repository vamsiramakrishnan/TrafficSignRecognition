
# coding: utf-8

# ### Import and Load Modules, Features from the Dataset
# #### Import Necessary Modules

# In[3]:

# skimage for Image Transformations
from skimage import filters
from skimage import color
from skimage import exposure
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
from skimage.transform import AffineTransform
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import cv2

# Matplotlib for Displaying Plots
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


# Random and Math and Numpy for Mathematical Operations
from random import randint
from random import uniform
import math
import numpy as np


# Pickle for Caching, Storing and Retrieving Data
import pickle

# Import Shuffling function from SKLEARN
from sklearn.utils import shuffle

# Pandas For Data Visualization, TQDM for Progress Bar
import pandas as pd
from tqdm import trange
from IPython.display import display, HTML

print("Modules Imported")


# #### Load the Pickle File 

# In[6]:

def load_data(file):
    with open(file, mode='rb') as f:
        file_ = pickle.load(f)  
    x_, y_ = file_['features'], file_['labels']
    print("Data and Modules loaded")
    return x_,y_


# ---
# 
# ## Step 1: Functions for Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.

# ### Summarizing the Spread and Characteristics of the dataset using Pandas
# 1. There are datsets that can be flippable and their meaning wouldnt essentially change.
# + There are signs which when flipped causes a change in their meaning. 
# + They are Summarized in a Pandas Data Frame.

# <img src="files/DataVisualization.png">

# <a id='Dataset Visualization Function'></a>
# #### Dataset Visualization Function

# In[5]:

def visualize_dataset(X_input, y_input, nr, nc):

    # Sort Images based on labels to batch the labels according to uniform size
    sorter = np.argsort(y_input)
    # Sort Dataset
    X_input = X_input[sorter]
    n_classes, counts = np.unique(y_input, return_counts=True)

    # TODO: Number of training examples
    n_train = len(X_input)
    # TODO: What's the shape of an traffic sign image?
    image_shape = X_input[0].shape
    df = pd.read_csv("signnames.csv")

    # Populating the Data Frame
    df['Counts'] = counts
    # Random Check to verify if the sorting of data is consistent.
    randval = randint(0, len(X_input) - 1)
    rowindex = y_input[randval]
    condition = df['ClassId'] == rowindex
    y_name = df[condition]

    # Classes of signs that, when flipped horizontally, should still be
    # classified as the same class
    self_flippable_horizontally = np.array(
        [11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    df_sfh = np.empty(len(n_classes), dtype=object)
    for i in range(len(df_sfh)):
        if i in self_flippable_horizontally:
            df_sfh[i] = "yes"
        else:
            df_sfh[i] = "no"
    df['Horizontally Flippable'] = df_sfh

    # Classes of signs that, when flipped vertically, should still be
    # classified as the same class
    self_flippable_vertically = np.array([1, 5, 12, 15, 17])
    df_sfv = np.empty(len(n_classes), dtype=object)
    for i in range(len(df_sfv)):
        if i in self_flippable_vertically:
            df_sfv[i] = "yes"
        else:
            df_sfv[i] = "no"
    df['Vertically Flippable'] = df_sfv

    # Classes of signs that, when flipped horizontally and then vertically,
    # should still be classified as the same class
    self_flippable_both = np.array([32, 40])
    df_sfb = np.empty(len(n_classes), dtype=object)
    for i in range(len(df_sfb)):
        if i in self_flippable_both:
            df_sfb[i] = "yes"
        else:
            df_sfb[i] = "no"
    df['Flippable Both Ways'] = df_sfb

    # Classes of signs that, when flipped horizontally, would still be
    # meaningful, but should be classified as some other class
    cross_flippable = np.array([[19, 20], [33, 34], [36, 37], [38, 39], [
                               20, 19], [34, 33], [37, 36], [39, 38]])
    df_cf = np.empty(len(n_classes), dtype=object)
    for i in range(len(df_cf)):
        if i in cross_flippable:
            z = np.nonzero(cross_flippable[:, 0] == i)
            df_cf[i] = df.ix[cross_flippable[z[0]][0][1], 'SignName']
        else:
            df_cf[i] = "no"
    df['CrossFlippable'] = df_cf

    ############ Display Section #################
    print("Number of training examples =", n_train)
    print("Image data shape =", image_shape)
    print("Number of classes =", len(n_classes))
    print("Image Datatype=", X_input.dtype)
    plt.rcParams['figure.figsize'] = (16, 6)
    display(df)
    plt.grid()
    plt.xlabel("Class -ID")
    plt.ylabel("Number of Images")
    plt.title("Data Histogram")
    plt.bar(n_classes, counts)
    images_show(X_input, nr, nc, rand=True)


def images_show(X_input, nr, nc, rand=True):
    
    # Show randomly chosen 100 images
    if (len(X_input.shape) == 4 and (X_input.shape[3] != 3)):
        X_disp = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], -1))
    else:
        X_disp = X_input

    if rand == True:
        randval1 = randint(0, len(X_disp) - 1)
    else:
        randval1 = 0

    randval2 = randval1 + (nr * nc)
    disp_im = X_disp[randval1:randval2]
    num_rows = nr
    num_cols = nc
    plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.1)
    ax = [plt.subplot(gs[i]) for i in range(num_rows * num_cols)]
    for index, index_im in enumerate(disp_im):
        ax[index].imshow(index_im, cmap='gray')
        ax[index].axis('off')
    plt.show()


# ## Step 2:  Functions for Augmentation to supplement the dataset
# The Augmentation algorithm is based on
# Data augmentation batch iterator for feeding images into CNN.
# 
# ###  Background Work
# <img src="files/DataPreProcessingSteps.png">
# 
# **For Rotation , Translation & Shearing**
# - **Rotate** all images in a given batch between -10 and 10 degrees.
# - **Random Translations** between -10 and 10 pixels in all directions.
# - **Random Zooms** between 1 and 1.3.
# - **Random Shearing** between -25 and 25 degrees.
# - randomly applies **Sobel Edge** detector to 1/4th of the images in each batch.
# - **Randomly Inverts** 1/4 of the images in each batch.
# 
# > **Source:   https://github.com/vxy10/ImageAugmentation**  - Vivek Yadav on Traffic Sign Classification.
# 
# > **Source:   http://florianmuellerklein.github.io/cnn_streetview/** - Florian Mieller on Street View using CNNs
# 
# **Mirroring & Flipping Function**:
# The function is used to augment the dataset with Zero Computational Effort
# The process followed is 
# - Identify classes that wouldn't change meaning when flipped
# - Identify classes that can generate another class when flipped. 
# - Make sure the labelling is done as per the change of meaning.  
# 
# > **Source:     http://navoshta.com/traffic-signs-classification **

# ### Functions for Image Normalization, Rotation , Augmentation , Flipping and Warping

# In[43]:

########## Extend Dataset by Flipping and Rotation  ###########
def flip_extend(X, y):
    # Classes of signs that, when flipped horizontally, should still be
    # classified as the same class
    self_flippable_horizontally = np.array(
        [11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    # Classes of signs that, when flipped vertically, should still be
    # classified as the same class
    self_flippable_vertically = np.array([1, 5, 12, 15, 17])
    # Classes of signs that, when flipped horizontally and then vertically,
    # should still be classified as the same class
    self_flippable_both = np.array([32, 40])
    # Classes of signs that, when flipped horizontally, would still be
    # meaningful, but should be classified as some other class
    cross_flippable = np.array([[19, 20], [33, 34], [36, 37], [38, 39], [
                               20, 19], [34, 33], [37, 36], [39, 38]])
    num_classes, counts = np.unique(y, return_counts=True)

    X_extended = np.empty(
        [0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    y_extended = np.empty([0], dtype=y.dtype)

    for c in range(len(num_classes)):
        # First copy existing data for this class
        X_extended = np.append(X_extended, X[y == c], axis=0)

        # If we can flip images of this class horizontally and they would still
        # belong to said class...
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(
                X_extended, X[y == c][:, :, ::-1, :], axis=0)

        # If we can flip images of this class horizontally and they would
        # belong to other class...
        if c in cross_flippable[:, 0]:
            # ...Copy flipped images of that other class to the extended array.
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(
                X_extended, X[y == flip_class][:, :, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full(
            (X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        # If we can flip images of this class vertically and they would still
        # belong to said class...
        if c in self_flippable_vertically:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(
                X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full(
            (X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        # If we can flip images of this class horizontally AND vertically and
        # they would still belong to said class...
        if c in self_flippable_both:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(
                X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis=0)

        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full(
            (X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

    return (X_extended, y_extended)

####### Image Normalizer for exposure based histogram equalization #######


def image_normalizer(X):

    norm = np.zeros(
        (X.shape[0], X.shape[1], X.shape[2], X.shape[3]), np.float32)
    norm_rgb = np.zeros(
        (X.shape[0], X.shape[1], X.shape[2], X.shape[3]), np.uint8)
    X = X * 255.0

    rgb = [123.68, 116.779, 103.939]

    for i in range(X.shape[0]):
        r = X[i, :, :, 0]
        g = X[i, :, :, 1]
        b = X[i, :, :, 2]

        norm[i, :, :, 0] = (r - rgb[0])
        norm[i, :, :, 1] = (g - rgb[1])
        norm[i, :, :, 2] = (b - rgb[2])
        
    return norm


# ### Data Augmentation and Perturbation Functions ##

# In[44]:

############## Invert the Image #####################
def invert(X, intensity=0.75, depth=1): 
    no_channels= X.shape[3]
    # invert half of the images
    indices_invert = np.random.choice(X.shape[0], math.ceil(X.shape[0] * depth), replace=False)
    X_=[]
    for l in indices_invert:
        img= X[l]
        for i in range(no_channels):
            img_=img[:,:,i]
            if img_.any()>0.5:
                min_distance= np.abs( img_- np.min(img))
                img[:,:,i]= - min_distance + np.amax(img_)
        np.clip(img, 0, 1, out=img)
        X_.append(img) 
        
    return np.asarray(X_)


########### Image Rotate Function ################
def img_rotate(X, intensity=0.75, depth=1):
    indices_rotate = np.random.choice(
        X.shape[0], math.ceil(X.shape[0] * depth), replace=False)
    delta = 30. * intensity
    X_=[]
    for i in indices_rotate:   
        X_.append(rotate(X[i],uniform(-delta, delta), mode='edge'))
    return np.asarray(X_)


############### Image Zoom Function ####################
def zoom(X, intensity=0.75, depth=1):
    image_size = X.shape[1]
    indices_zoom = np.random.choice(X.shape[0], math.ceil(X.shape[0] * depth * 0.5), replace=False)
    X_=[]
    for k in indices_zoom:
        zoom_fac= intensity/(1.5)
        zoom_x= uniform(1 - zoom_fac, 1 + zoom_fac)
        zoom_y= uniform(1 - zoom_fac, 1 + zoom_fac)

        transform= AffineTransform(scale=(zoom_x, zoom_y))
        X_.append(warp(X[k], transform.inverse, output_shape=(
                image_size, image_size), order=1, mode='edge'))
        X_.append(warp(X[k], transform, output_shape=(
                image_size, image_size), order=1, mode='edge'))
    return np.asarray(X_)


################# Apply a Gaussian Blur ##############
def gaussian(X, intensity=0.75, depth=1):
    indices_gaussian = np.random.choice(
        X.shape[0], math.ceil(X.shape[0] * depth), replace=False)
    X_=[]
    for k in indices_gaussian:
        sigma_=uniform(1-intensity,intensity)
        X_.append(filters.gaussian(X[k], sigma=sigma_, multichannel=True))
    return np.asarray(X_)

    
################ Adaptive Histogram equalization ############
def ahisteq(X, intensity=0.75, depth=1):
    indices_ahisteq = np.random.choice(
        X.shape[0], math.ceil(X.shape[0] * depth), replace=False)

    X_=[]
    for k in indices_ahisteq:
        X_rgb=X[k]
        X_rgb[:,:,0] = exposure.equalize_hist(X_rgb[:, :, 0])
        X_rgb[:,:,1] = exposure.equalize_hist(X_rgb[:, :, 1])
        X_rgb[:,:,2] = exposure.equalize_hist(X_rgb[:, :, 2])
        X_.append(X_rgb)
    
    return np.asarray(X_)
    
################# Histogram Equalization ######################
def histeq(X, intensity=0.75, depth=1):
    # Apply histogram equalization on one quarter of the images
    indices_histeq = np.random.choice(
        X.shape[0], math.ceil(X.shape[0] * depth), replace=False)

    X_=[]
    for k in indices_histeq:
        X_rgb=X[k]
        X_rgb[:,:,0] = exposure.equalize_hist(X_rgb[:, :, 0])
        X_rgb[:,:,1] = exposure.equalize_hist(X_rgb[:, :, 1])
        X_rgb[:,:,2] = exposure.equalize_hist(X_rgb[:, :, 2])
        X_.append(X_rgb)
    
    return np.asarray(X_)


############## adapthisteq ######################################
def gamma(X, intensity=0.75, depth=1):
    # Apply Gamma on one quarter of the images
    indices_gamma = np.random.choice(
        X.shape[0], math.ceil(X.shape[0] * depth), replace=False)
    X_=[]
    for k in indices_gamma:
        gamma_=uniform(1 - intensity , 1 + intensity )
        X_.append(exposure.adjust_gamma(X[k], gamma_))
    return np.asarray(X_)


################# Random increment of brightness ######
def augment_brightness(X, intensity=0.75, depth=1):
    X = np.asarray([color.rgb2hsv(img) for img in X])
    indices_randbright = np.random.choice(
        X.shape[0], math.ceil(X.shape[0] * depth), replace=False)
    X_=[]
    for k in indices_randbright:
        random_bright= np.random.uniform(-intensity / 3, intensity / 3)
        img_1 = X[k]
        img_1[:, :, 2] = img_1[:, :, 2] + random_bright
        img_1[:, :, 2][img_1[:, :, 2] > 255] = 255
        X_.append(img_1)

    return np.asarray([color.hsv2rgb(img) for img in X_])


#######For Affine , Shear, Scale and Rotation, Projective Transform ################
def apply_projection_transform(X, intensity=0.75, depth=1):
    no_samples=X.shape[0]
    image_size=X.shape[1]
    no_channels=X.shape[3]
    d = image_size * 0.3 * intensity
    indices_project = np.random.choice(
        X.shape[0], math.ceil(X.shape[0]*depth*0.5), replace=False)
    X_=[]              
    for i in indices_project:
        tl_top = uniform(-d, d)     # Top left corner, top margin
        tl_left = uniform(-d, d)    # Top left corner, left margin
        bl_bottom = uniform(-d, d)  # Bottom left corner, bottom margin
        bl_left = uniform(-d, d)    # Bottom left corner, left margin
        tr_top = uniform(-d, d)     # Top right corner, top margin
        tr_right =uniform(-d, d)   # Top right corner, right margin
        br_bottom =uniform(-d, d)  # Bottom right corner, bottom margin
        br_right = uniform(-d, d)   # Bottom right corner, right margin

        transform = ProjectiveTransform()
        transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))

        X_.append(warp(X[i], transform, output_shape=(image_size, image_size), order = 1, mode = 'edge'))
        X_.append(warp(X[i], transform.inverse, output_shape=(image_size, image_size), order = 1, mode = 'edge'))
        
    return np.asarray(X_)  

############# Append Labels to increase size of label set  ##############################
def inc(Y, depth_):
    indices_ = np.random.choice(
        Y.shape[0], math.ceil(Y.shape[0] * depth_), replace=False)
    Y_ = []
    for i in indices_:
        Y_.append(Y[i])
    return Y_


# ### Batch Iteration and Data Augmentation Helper functions ###

# In[45]:


def Augment_Images(X, Y, intensity_factor, same_size=False):
    # Intensity defines the rate at which the Images are transformed
    # Rotate, Shear and Scale all images

    sequential_depth = 0.75
    depth_ = 1 - sequential_depth

    if not same_size:

        ############ Random Brightness ############
        X_a = augment_brightness(X, intensity_factor, depth_)
        Y_a = inc(Y, depth_)

        ########## Histogram Equalization ###########
        X_h = histeq(X, intensity_factor, depth_)
        Y_h = inc(Y, depth_)

        X_ah = ahisteq(X, intensity_factor, depth_)
        Y_ah = inc(Y, depth_)

        ############ Rotations ######################
        X_r = img_rotate(X, intensity_factor, depth_)
        Y_r = inc(Y, depth_)

        ############# Zoom ###########################
        X_z = zoom(X, intensity_factor, depth_)
        Y_z = inc(Y, depth_)

        ########## Shear #################################
        X_p = apply_projection_transform(X, intensity_factor, depth_)
        Y_p = inc(Y, depth_)

        ########### Gaussian Noise ########################
        X_g = gaussian(X, intensity_factor, depth_)
        Y_g = inc(Y, depth_)

        X_i = invert(X, intensity_factor, depth_)
        Y_i = inc(Y, depth_)

        ############# Sequentially apply all ##############
        X_seq = augment_brightness(X, 0.75, sequential_depth)
        X_seq = histeq(X_seq)
        X_seq = img_rotate(X_seq)
        X_seq = apply_projection_transform(X_seq)
        X_seq = gaussian(X_seq)
        Y_seq = inc(Y, sequential_depth)

        ############## Concatenate all results ###################
        X_ = np.concatenate(
            (X_a, X_h, X_ah, X_p, X_r, X_z, X_i, X_seq), axis=0)
        Y_ = np.concatenate(
            (Y_a, Y_h, Y_ah, Y_p, Y_r, Y_z, Y_i, Y_seq), axis=0)

    else:
        ############# Sequentially apply all ##############
        X_seq = augment_brightness(X, intensity_factor, 1)
        X_seq = histeq(X_seq)
        X_seq = img_rotate(X_seq)
        X_seq = zoom(X_seq)
        X_seq = apply_projection_transform(X_seq)
        X_seq = gaussian(X_seq)
        X_seq = ahisteq(X_seq)
        Y_seq = inc(Y, 1)
        X_ = X_seq
        Y_ = Y_seq

    return X_.astype(np.float32), Y_


############################ Create a balanced dataset of given sample siz
def bal_dataset(X, Y, sample_size):
    offset = 0
    sorter_ = np.argsort(Y)
    
    # Sort Dataset
    Y = Y[sorter_]
    X = X[sorter_]
    n_classes, counts = np.unique(Y, return_counts=True)
    offset = 0
    offset_= 0
    
    X_ = np.zeros((int(sample_size*len(n_classes)),X.shape[1],X.shape[2],X.shape[3]),dtype=np.float32)
    Y_ = np.zeros(int(sample_size*len(n_classes),),dtype=np.int32)
    SAMPLE_SIZE = sample_size - 1
    print("Balancing Dataset by truncating data at random")
    for i in trange(len(n_classes)):
        BATCH_SIZE = counts[i] - 1
        
        batch_X, batch_Y = X[offset:offset + BATCH_SIZE], Y[offset:offset + BATCH_SIZE]
        batch_X, batch_Y = shuffle(batch_X, batch_Y)
    
        X_[offset_: offset_+SAMPLE_SIZE] = batch_X[0:SAMPLE_SIZE]
        Y_[offset_: offset_+SAMPLE_SIZE] = batch_Y[0:SAMPLE_SIZE]
        
        ###### Offset ###########
        offset += BATCH_SIZE + 1
        offset_+= SAMPLE_SIZE + 1
        
    return np.asarray(X_), np.asarray(Y_)


############################## Iterate the samples batch wise ############
def batch_iterator(X, Y, sample_size, intensity_factor, balance_dataset):
    n_classes, counts = np.unique(Y, return_counts=True)
    offset = 0

    for i in trange(len(n_classes)):
        BATCH_SIZE = counts[i] - 1
        print("Augmenting Data for Class Number:",n_classes[i])
        # Augmentation factor is currently scaled based on
        # number of samples required to match the class with max samples.
        aug_fac = (math.ceil(sample_size / BATCH_SIZE)) - 1
        batch_X, batch_Y = X[offset:offset + BATCH_SIZE], Y[offset:offset + BATCH_SIZE]

        
        # Use the batch iterator from previously
        # defined function to create Datasets
        for j in range(aug_fac):
            batch_X, batch_Y = shuffle(batch_X, batch_Y)
            X_aug, Y_aug = Augment_Images(batch_X, batch_Y, intensity_factor, same_size=True)
            X = np.append(X, X_aug, axis=0)
            Y = np.append(Y, Y_aug, axis=0)
        offset+= BATCH_SIZE + 1 
        
    if balance_dataset is True:
        X, Y = bal_dataset(X, Y, sample_size)

    return X, Y


# #### Scale Dataset and Extend Dataset

# In[46]:

def scale_dataset(X,Y ,visualize=True, showimages=False):
    X_ = (X / 255.-.999).astype(np.float32)
    if visualize:
        visualize_dataset(X_, Y, 10, 10)
    elif showimages:
        images_show(X_,Y, 10, 10)

    return X_,Y

def extend_dataset(X,Y,visualize=True, showimages=False):
    X_, Y_ = flip_extend(X,Y)
    if visualize:
        visualize_dataset(X_, Y_, 10, 10)
    elif showimages:
        images_show(X_,Y_)
        
    return X_,Y_


# ### Augmentation Process
# - Revisualization
# - Re- Plotting the dataset post augmentation

# In[7]:

def augment_data(X, Y, sample_size, intensity_factor, is_balance, visualize=True, showimages=False):
    
    sorter = np.argsort(Y)
    # Sort Dataset
    Y = Y[sorter]
    X = X[sorter]
    
    X_, Y_ = batch_iterator(
        X, Y, sample_size, intensity_factor, balance_dataset=is_balance)
    
    
    # Make sure the augmented data set is a number divisible by 100 for flexible 
    # batching
    mod = len(X_) % 100
    trunc = 100 - mod
    index_range = np.arange(0, trunc, 1)
    print("Making the dataset divisible by 100 by adding images")
    for j in index_range:
        rand_index = randint(0, len(X_) - 1)
        temp1 = np.expand_dims(X_[rand_index], axis=0)
        temp2 = np.expand_dims(Y_[rand_index], axis=0)
        X_ = np.append(X_, temp1, axis=0)
        Y_ = np.append(Y_, temp2, axis=0)
  
    
    # Randomly Display 100 images in a given class to
    # see the output of augmentation
    print("New Dataset Size:", len(X_))
    X_, Y_ = shuffle(X_, Y_)
    if visualize:
        visualize_dataset(X_, Y_, 10, 10)
    elif showimages:
        images_show(X_,Y_)
        
    return X_, Y_


# In[11]:

def cache_data(X,Y, file):
    n_samples=X.shape[0]
    try:
            with open(file, 'wb') as pfile:
                pickle.dump(
                    {
                        'features': X.astype(np.float32),
                        'labels': Y
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
                print("Data Saved in :",file)

    except Exception as e:
        print('Unable to save data to a single file so splitting data into 3', file, ':', e)
        with open(file+'_1', 'wb') as pfile:
            pickle.dump(
                {
                    'features': X[0:int((1/3)*n_samples)].astype(np.float32),
                    'labels': Y[0:int((1/3)*n_samples)]
                },
                pfile, pickle.HIGHEST_PROTOCOL)

        with open(file+'_2', 'wb') as pfile:
            pickle.dump(
                {
                    'features': X[int(n_samples/3):int((2/3)*n_samples)].astype(np.float32),
                    'labels': Y[int(n_samples/3):int((2/3)*n_samples)]
                },
                pfile, pickle.HIGHEST_PROTOCOL)

        with open(file+'_3', 'wb') as pfile:
            pickle.dump(
                {
                    'features': X[int((2/3)*n_samples):n_samples-1].astype(np.float32),
                    'labels': Y[int((2/3)*n_samples):n_samples-1]
                },
                pfile, pickle.HIGHEST_PROTOCOL)
        
        print("Data Saved in :",file)
        print('pickle file saved as 3 parts for data')      


# In[ ]:

def preprocess_data(source, target, is_scale=True, is_extend=False, is_augment=False, sample_size=1000, intensity_factor=0.5, is_balance=True):
    
    X_,Y_= load_data(source)  
    ########### Scale the dataset by default ########
    if is_scale:
        print("Scaling Start")
        print("Scaling all features between -1 and 1")
        X_,Y_ = scale_dataset(X_,Y_,visualize=False,showimages=False)
        print("Scaling Complete")
    if is_extend:
        print("Extending Dataset")
        X_,Y_ = extend_dataset(X_,Y_,visualize=False,showimages=False)
        print("Dataset Extended based on Flipping, Mirroring 0-180 Degrees")
    if is_augment:
        print("Data Augmentation Started")
        X_,Y_ = augment_data(X_, Y_, sample_size, intensity_factor, is_balance, visualize=True, showimages=False)
    else:
        visualize_dataset(X_,Y_,10,10)
        
    cache_data(X_,Y_, target)
         
