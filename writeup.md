# **Behavioral Cloning** 
## Self Driving Car Nano Degree
### Term : 1,  Project : 4
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/mse_lr_002_ep_10.png "Mean square error"
[image6]: ./examples/original_and_flipped.png "original_and_flipped.png"


## Rubric Points

### Here the [rubric points](https://review.udacity.com/#!/rubrics/432/view) are considered individually, and described how they are addressed in this implementation.  

---
### CATEGORY : Required Files


#### CRITERIA 1 : Are all required files submitted?

This repository includes the following files:
* model.py containing the script to create and train the model
* model.h5 containing a trained convolution neural network 
* drive.py for driving the car in autonomous mode
* writeup.md summarizing the results
* video.mp4 showing how the above saved model operates the car in autonomous mode.
* and the Extras folder, that contains the source code in .ipynb file, so the entire implementation can be viewed on github.

---
### CATEGORY : Quality of Code


#### CRITERIA 1 : Is the code functional?
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Note that this drive.py file has been modified to operate the car in autonomous mode at 15 kmph.


#### CRITERIA 2 : Is the code usable and readable?

* The model.py file contains the code for training and saving the convolution neural network. The file shows the implementation used for training and validating the model, and it contains comments to explain how the code works.

> A `generator factory method` has been used, which returns `generator functions` for training and validation. This is used in fit_generator method, for avoiding usage of all the available memory. Here, the images are loaded into the memory only one batch at a time.

* The **augmented images** and **left & right camera pictures** are also considered in this generator functions, based on the flags set during the creation of this functions from the factory method (flags are set for training generator, not for validation generator)

* The correction factor used for steering angles on pictures from left and right camera are 0.3 and -0.3 respectively.

```python
# Augmenting a data with cv2 flip.
center_image_flipped = np.fliplr(center_image)
center_angle_flipped = -center_angle
images.append(center_image_flipped)
angles.append(center_angle_flipped)
```

---
### CATEGORY : Model Architecture and Training Strategy


#### CRITERIA 1 : Has an appropriate model architecture been employed for the task?

* The model consists of a convolution neural network, as proposed by nvidia in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 

* The entire model is architectured within one api, `create_nvidia_model`, which returns a fully structured and compiled model.

* Along with the proposed architecture, 
  - the data is normalized in the model using a Keras lambda layer
  - a cropping layer is used to remove unwanted pixels (sky and hood), using a Keras lambda layer
  - the model includes RELU layers to introduce nonlinearity,
  - dropouts are added after each convolution and dense layers, to avoid overfitting on training data.
  
Note : Detailed design of the architecture of model will be discussed in later criterias.

* The final output is a single continuous value, as this is a linear regression and not a classification problem.

* Hence, only `mse` (Mean Square Error) is calculated, and not the accuracy.


#### CRITERIA 2 : Has an attempt been made to reduce overfitting of the model?

* The model contains dropout layers in order to reduce overfitting, as designed in the `create_nvidia_model` function.

* The Train / Validation data set split is made using `train_test_split` function from `sklearn.model_selection`, in the following ratio
  - Training set   : 80%
  - Validation set : 20%


#### CRITERIA 3 : Have the model parameters been tuned appropriately?

* Even though the optimizer used is Adam optimizer, the default initial learning rate is very less (0.001).

* As we use only few epochs, it is better to set a learning rate manually, so that the model converges faster.

* The training set mse loss took many epochs to reduce, whereas the validation set mse loss reduced to a much lesser value right from initial epoch.

* Hence, different learning rates were experimented: (epoch : 10)
  - Learning rate : 0.001     Training set loss : 0.0220      Validation set loss : 0.0210
  - Learning rate : 0.002     Training set loss : 0.0222      Validation set loss : 0.0188
  - Learning rate : 0.003     Training set loss : 0.0245      Validation set loss : 0.0237
  - Learning rate : 0.005     Training set loss : 0.0290      Validation set loss : 0.0241
  - Learning rate : 0.008     Training set loss : 0.0292      Validation set loss : 0.0332
  
> The main reason for training set loss being higher than validation set loss is that the training set generator produced images for center, left and right camera images, and aslo augmented images for the entire image set.

> Hence, for **each and every batch size of 32**, the training generator produced 
```math
32 * 3(center, left, right) * 2(augmented) = 192 images
```

> Whereas, for a batch size of 32 in validation set produced only the real 32 image set, with no arbitrary values.

![alt text][image1]

* Finally , a learning rate of 0.002 and and epoch 10 is choosen.

#### CRITERIA 4 : Is the training data chosen appropriately?


* Augmented images are produced using cv2 flip, for each and every image in the training set. This helps in generalising the model.

* For each and every image, corresponding left and right camera images are used, with the steering angle corrected by a factor of (+/-)3, and is fed to the model for training, as an image from center image.

* This greatly reduces the monopoly of images with steering angle 0. 

* This also helps in keeping the car on track, if it veers of to any side. These steering angles are deduced based on only the center camera image during actual simulation. Hence, feeding the left / right images helps the model in taking better decisions

* As the driving lane is a loop, many left turns are present, which increases the bias on left steering angles upon training the model. By augmenting the image and steering angle by a vertical flip rectifies this problem.

---
### CATEGORY : Architecture and Training Documentation


#### CRITERIA 1 : Is the solution design documented?

* As discussed in the class, **[Nvidia's Covnet](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)** is used as the starting point for designing this cnn architecture.

* But, the paper did not discuss anything about dropout or any other regularization techniques.

* Hence, to generalize the cnn, few droupouts and relu layers are added afetr the certain cnn and dense layers (discussed as a part of next criteria).

* The Dropout probabilities are altered after several experimentations, to avoid fluctuations in the training loss, and also to avoid overfitting.

* A **Keras lambda layer** is used to normalize the image frames received, and an other lambda layer for croping out the unwanted pixels (sky and hood). 

* These should be done as a part of the model, and should not be preprocessed, as the images are directly fed to the model during the actual simulation of autonomous mode.

#### CRITERIA 2 : Is the model architecture documented?

* The characteristics and qualities of the architecture are already described as a part of previous categories.

* The final model architecture, as designed in function `create_nvidia_model` consists of a convolution neural network with the following layers and layer sizes :
  
| Layer         | Layer Specs                                | Output Size |
|---------------|--------------------------------------------|-------------|
| Normalization | `lambda x: (x-128) / 128`                  | 160x320x3   |
| Cropping      | Cropping2D(cropping=((70, 25), (0, 0))     | 65x320x3    |
| Convolution   | 24, 5x5 kernels, 2x2 stride, valid padding | 31x158x24   |
| RELU          | Non-linearity                              | 31x158x24   |
| Convolution   | 36, 5x5 kernels, 2x2 stride, valid padding | 14x77x36    |
| RELU          | Non-linearity                              | 14x77x36    |
| Convolution   | 48, 5x5 kernels, 1x1 stride, valid padding | 5x37x48     |
| RELU          | Non-linearity                              | 5x37x48     |
| Dropout       | Probabilistic regularization (p=0.1)       | 5x37x48     |
| Convolution   | 64, 3x3 kernels, 1x1 stride, valid padding | 3x35x64     |
| RELU          | Non-linearity                              | 3x35x64     |
| Dropout       | Probabilistic regularization (p=0.1)       | 3x35x64     |
| Convolution   | 64, 3x3 kernels, 1x1 stride, valid padding | 1x33x64     |
| RELU          | Non-linearity                              | 1x33x64     |
| Dropout       | Probabilistic regularization (p=0.1)       | 1x33x64     |
| Flatten       | Convert to vector.                         | 2112        |
| Dense         | Fully connected layer. No regularization   | 100         |
| Dropout       | Probabilistic regularization (p=0.3)       | 100         |
| Dense         | Fully connected layer. No regularization   | 50          |
| Dropout       | Probabilistic regularization (p=0.3)       | 50          |
| Dense         | Fully connected layer. No regularization   | 10          |
| Dense         | Output prediction layer.                   | 1           |

(note: visualizing the architecture is optional according to the project rubric)


#### CRITERIA 3 : Is the creation of the training dataset and training process documented?

As described in th eprevious criterias:

* The manual driving mode data (simulation images and steering angle) provided by udacity in the workspace was used initially. But the data is not sufficient, and hence more data was generated.

* Augmented images are produced using cv2 flip, for each and every image in the training set. This helps in generalising the model.

* For each and every image, corresponding left and right camera images are used, with the steering angle corrected by a factor of (+/-)3, and is fed to the model for training, as an image from center image.

* This greatly reduces the monopoly of images with steering angle 0. 

* This also helps in keeping the car on track, if it veers of to any side. These steering angles are deduced based on only the center camera image during actual simulation. Hence, feeding the left / right images helps the model in taking better decisions

* As the driving lane is a loop, many left turns are present, which increases the bias on left steering angles upon training the model. By augmenting the image and steering angle by a vertical flip rectifies this problem.

* The generator function takes care of batching and generating data for training :

Totally, 8036 samples were availbale
```math
These are divided into : 
Training set   : 6428
validation set : 1608
```

The training set is used to deduce left and right steering angles :
```math
Training set size : 6428 * 3 = 19284
```

And all these images are augmented :
```math
Training set size : 19284 * 2 = 38568
```

![alt text][image6]


---
### CATEGORY : Simulation

#### CRITERIA 1 : Is the car able to navigate correctly on test data?

* The model drives the car successfully in the autonomous mode of the simulation.

* A [video playback](./video.mp4) has been generated with the lane frames grabbed from autonomous mode, using video.py


---
### Possible areas of improvement:
* More data can be generated, to make the model robust.
* The Nvidia's paper suggest YUV colorspace, but in tis simulation, RGB colorspace gives a better accuracy.
* Hence, a better filtering mehchanism for the frames should ben identified.
* The car shall be trained on second lane, so that a robust and generalized model that can be achieved.
