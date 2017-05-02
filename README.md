# **README**

## **Traffic Sign Recognition** 

### **Victor Roy**

[project code](https://github.com/soniccrhyme/SDND-Project_2)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./graphics/train-count_boxplot.png "Count in Train Boxplot"
[image2]: ./graphics/rand-images.png "Sample of Images"
[image3]: ./graphics/model_architecture.png "Model Architecture"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

--

### Data Set Summary & Exploration

#### 1. Dataset summary:

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 pixels
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a boxplot showing the distribution of images in each of the classes. Some classes are represented by as few as 200 examples while others have nearly if not more than 2000. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Augmenting and Preprocessing Data

The dataset has uneven representation across classes and not as many examples as might be required to train a successful model. It thus seemed prudent to augment the dataset. 

Some image classes, such as 'General Caution' or 'Priority Road' could be flipped across the vertical access and still be part of the same class. For images in other classes, such as 'Keep Left/Right' or 'Turn left/right ahead' could be similarly flipped with the result belonging to a complementary class. Inspiration for this was taken from [Alex Staravoitau's work](http://navoshta.com/traffic-signs-classification/).

Additional data was created by adding some random rotation and scaling factors to existing images. More images were created for less well-represented classes. The goal was to get at least 2000 images per class. By using the scikit-image library, I was able to transform each image into *n* number of additional images by randomly adjusting scale, rotation and gamma (i.e. brightness). 

The resultant dataset had 223,760 images. 

This dataset was then preprocessed. Given the results in [Sermanet & Lecun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), it seemed unneccessary to use all three channels, so images were translated into grayscale using scikit-image. Further, the contrast for many of the images seemed to vary a lot - the range of pixel values represented were not constant. Some images were far brighter, other far darker. To remedy this variation, I tried using three different methods within [scikit-image's exposure class](http://scikit-image.org/docs/dev/api/skimage.exposure.html): rescale\_intensity, equalize\_hist, and equalize\_adapthist. After some trials, the equalize\_adapthist method seemed to provide the best results. 

Finally, the data was scaled between [-0.5,+0.5] by subtracting and then dividing by 128. This seemed like a reasonable simple method to employ because the equalize\_adapthist method has the effect of standardizing the range of pixel values represented in an image.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The architecture for my model draws inspiration from [Sermanet & Lecun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) as well as [Alex Staravoitau's work](http://navoshta.com/traffic-signs-classification/) and is detailed in the graphic below:


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


