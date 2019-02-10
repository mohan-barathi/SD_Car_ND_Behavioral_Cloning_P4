# Behavioral Cloning Project
## Self Driving Car Nano Degree
### Term : 1 Project : 4

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains the source code and output of Behavioral cloning project.

Here, deep neural networks and convolutional neural networks are used to clone driving behavior. This is implemented using Keras, to predict steering angle of an autonomous vehicle simulation.

A detailed discussion about the implementation of this model is availbale in the report writeup. This repository contains the following files:

* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* a report writeup file (markdown)
* video.mp4 (a video recording of the vehicle driving autonomously)

Important Links :
---
* [Rubric Points](https://review.udacity.com/#!/rubrics/432/view)
* [Source code for model](./model.py)
* [Saved Model](./model.h5)
* [Writeup](./writeup.md)
* [Output Video](./video.mp4)
* [as a notebook on nbviewer](https://nbviewer.jupyter.org/github/mohan-barathi/SD_Car_ND_Behavioral_Cloning_P4/blob/master/Extras/SDCar-ND_Project4-Behavioral_Cloning.ipynb)

Dependencies :
---
* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
  - Keras : 2.0.9
  - TensorFlow-GPU : 1.1.0
* [Car Simulation - Windows](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### To Execute the simulation in autonomous mode, using this saved model

### Run `drive.py`

`drive.py` requires trained model as an h5 file, i.e. `model.h5`.

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```
The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

