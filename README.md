## **README**
---

## **Traffic Sign Recognition Classifier using Convolutional Neural Nets**

### **Victor Roy**

[GitHub Link](https://github.com/soniccrhyme/SDND-Project_2)

---


[//]: # (Image References)

[image1]: ./graphics/train-count_boxplot.png "Count in Train Boxplot"
[image2]: ./graphics/rand-images.png "Sample of Images"
[image3]: ./graphics/model_architecture.png "Model Architecture"
[image4]: ./graphics/custom_roadsigns.png "German Traffic Signs taken from Google Streetview"
[image5]: ./graphics/custom_roadsigns_predictions.png "Predictions for Traffic Signs taken from Google Streetview"
[image6]: ./german_roadsigns/test_sign_3.png "Traffic Sign 3"
[image7]: ./german_roadsigns/test_sign_4.png "Traffic Sign 4"
[image8]: ./german_roadsigns/test_sign_5.png "Traffic Sign 5"

---

### I. Data Set Summary & Exploration

#### 1. Dataset summary:

* The size of training set contains 34,799 images
* The size of the validation set contains 4,410 images
* The size of test set contains 12,630 images
* The shape of a traffic sign image is 32x32 pixels
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a boxplot showing the distribution of images in each of the classes; some classes are represented by as few as 200 examples while others have nearly, if not more, than 2000:

![alt text][image1]

Here is an example of some randomly selected images and their respective classes:
![alt text][image2]

---

### II. Design and Test a Model Architecture

#### 1. Augmenting and Preprocessing Data

The dataset has uneven representation across classes and not as many examples as might be required to train a successful model. It thus seemed prudent to augment the dataset.

Some image classes, such as 'General Caution' or 'Priority Road' could be flipped across the vertical access and still be part of the same class. For images in other classes, such as 'Keep Left/Right' or 'Turn left/right ahead' could be similarly flipped with the result belonging to a complementary class. Inspiration for this was taken from [Alex Staravoitau's work](http://navoshta.com/traffic-signs-classification/).

Additional data was created by adding some random rotation and scaling factors to existing images ('perturbation' or 'jittering'). More images were created for less well-represented classes. The goal was to get at least 2000 images per class. By using the scikit-image library, I was able to transform each image into *n* number of additional images by randomly adjusting scale, rotation and gamma (i.e. brightness).

The resultant dataset contains 223,760 images.

This dataset was then preprocessed. Given the results in [Sermanet & Lecun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), it seemed unneccessary to use all three channels, so images were translated into grayscale using scikit-image. Further, the contrast for many of the images seemed to vary a lot - the range of pixel values represented were not constant across images. Some images were far brighter, others far darker. To remedy this variation, I tried using three different methods within [scikit-image's exposure class](http://scikit-image.org/docs/dev/api/skimage.exposure.html): rescale\_intensity, equalize\_hist, and equalize\_adapthist. After some trials, the equalize\_adapthist method seemed to provide the best results.

Finally, the data was scaled between [-0.5,+0.5] by subtracting and then dividing by 128. This seemed like a reasonable method to employ because the equalize\_adapthist method has the effect of standardizing the range of pixel values represented in an image.


#### 2. Model Architecture

The architecture for my model draws inspiration from [Sermanet & Lecun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) as well as [Alex Staravoitau's work](http://navoshta.com/traffic-signs-classification/) and is detailed in the graphic below:

![alt text][image3]

The model utilizes 3 convulutional neural net layers as well as 2 fully connected neural nets. Following Sermanet & Lecun, the outputs from the three CNNs are also used as inputs into the first fully connected layer. Furthermore, ELU was used as the activation function; this was chosen based on the improvements given by ELU over ReLU as depicted in [Mishkin et al (2016)](https://arxiv.org/pdf/1606.02228.pdf) & [Clevert et al (2016)](https://arxiv.org/abs/1511.07289).

The code for the architecture is laid out in *In[22]*. 

#### 3. Model Training, (Hyper)Parameter Selections

The model was tested using the following sets of hyperparameters & optimizers; those utilized are in **bold**:
* Optimizers: [MomentumOptimizer, **AdamOptimizer**]
* Batch\_sizes: [64, 128, 192, **256**]
* Learning\rates: [0.1, 0.01, 0.005, 0.001, **0.0005**, 0.0001]
* Epochs: variable, depending on when the model seemed to stop improving the prediction accuracy on the validation set
* Activation Functions: [ReLU, **ELU**]
* Dropout: [Not Used; Used, where keep\_prob: [0.75, 0.5], [1.0, 0.5], [0.75, 0.5], [1.0, 0.75]]

Ideally (i.e. as in scikit-learn) I would be able to utilize a GridSearchCV or RandomizedSearchCV wrapper to test various combinations of hyperparameters. Since this option wasn't readily available, I manually tested various combinations of hyperparameters as well as activation and regularization methods. For the learning\_rate I used a coarse-to-fine search, where I tried factors of 10 and then, upon finding one that worked best, tried to hone in on the most appropriate learning\_rate. This was especially time consuming because adjusting different filters, the number of layers or the optimizer would change the scope of the most effective learning rate. 


#### 4. Model Training, Discussion

Final model results:
* training set accuracy of 99.2%
* validation set accuracy of 97.3%
* test set accuracy of 94.5%

Training and Validation accuracies were calculated in *In/Out[26]* as the model was being trained. The Test set accuracy was calculated in *In/Out[27]

I first made sure the base LeNet model yielded a validation and test set accuracy of ~89%. After that, I was unclear of how to go about improving the architecture, but knew that it needed to be able to handle the added complexity of dealing with 43 classes of traffic signs rather than only 10 different handwritten digits.

I turned to a forum post which led me to several blog posts as well as a few papers (referenced below). It seemed like a multi-scale cnn approach was garnering success, so I decided to implement such an architecture. As opposed to normal ConvNets, multi-scale CNNs pass through outputs from different CNN layers to other layers furhter down in the architecture. In this case, outputs from the three CNN layers were fed into a final, fully-connected neural network. 

This particular architecture seems rather amenable to image classification because of its chained and multi-scale implementation of CNNs, in addition to other regularization and activation methods. CNNs provide a powerful way of parsing an image into patterns of features. Three levels of CNN layers were chosen such that different scopes of an image's details and characteristics would have a chance to be represented in the model (i.e. edges, shapes, letters and larger patterns). This type of architecture's success also seemed to be well documented in various references (see below). 

After a base architecture was chosen (which I named LeSermaNet after Sermanet & Lecun) I tried different values for the various hyperparameters (outlined in #3 above) as well as CNN filter and NN depths .

The depth of the various levels were chosen rather arbitrarily but seemed to work out well enough. 

The model does overfit a bit in later epochs - once the accuracy of the validation set reaches ~97%; therefore, it may do well to reduce the model's complexity some or add some additional regularization (either via dropout for the CNN layers or via implementation of l2\_regularization). Either that, or it might help to generate even more data to see if that helps stem overfitting.

---

### III. Test a Model on New Images

#### 1. Five Arbitrarily Chosen German Traffic Signs

Here are five German traffic signs that I grabbed from screenshots of Google Streetview in the town of Nuremburg (in honor of the Nurburgring; afterall this is a class about cars).

![alt text][image4]

Some observations: the 'General Caution', 'Priority Road' and 'Speed limit (30 km/h)' signs all have other signs adjacent to them; the 'Keep Right' sign is taken from an angle, so the sign seems more ovular; the 'Speed limit (30 km/h)' sign also has some shadows strewn across its left-side. All these factors may detract from the models ability to successfully classify each image.

#### 2. Model Predictions for the Five Traffic Signs from Google Streetview

Here are the results of the prediction:

![alt text][image5]

The model was able to correctly guess 5 of the 5 traffic signs, a perfect 100% accuracy rate. Classification of this small arbitrarily chosen sample outperforms that of the test set. I imagine that these images were relatively more easy to classify than others in the test set because of their source. Google generates its Streetview models on days which seem to be clear and bright. The visibility of all the signs was thus better than many of the other images contained within the dataset. 

#### 3. Softmax Probabilities for the Five Traffic Signs from Google Streetview 

Here are the Softmax-derived probabilities of the top 5 classes. As you can see, the model dealt with these categories pretty well, with > 99.9% confidence for each of the images. There must be other classes this model does less well with, possibly correlated with classes less well represented in the training set. I would create a bar chart, but it doesn't seem as if those charts would readily communicate any further information. 

Actual Class for Sign - 18: General caution
                           Class  Probability
General caution               18       0.9997
Pedestrians                   27       0.0003
Road narrows on the right     24       0.0000
Traffic signals               26       0.0000
Double curve                  21       0.0000


Actual Class for Sign - 38: Keep right
                      Class  Probability
Keep right               38          1.0
Priority road            12          0.0
Turn left ahead          34          0.0
Yield                    13          0.0
Roundabout mandatory     40          0.0


Actual Class for Sign - 12: Priority road
                                     Class  Probability
Priority road                           12          1.0
Keep right                              38          0.0
End of all speed and passing limits     32          0.0
Yield                                   13          0.0
No entry                                17          0.0


Actual Class for Sign - 17: No entry
                   Class  Probability
No entry              17          1.0
Priority road         12          0.0
End of no passing     41          0.0
Yield                 13          0.0
Stop                  14          0.0


Actual Class for Sign - 1: Speed limit (30km/h)
                      Class  Probability
Speed limit (30km/h)      1          1.0
Speed limit (80km/h)      5          0.0
Speed limit (50km/h)      2          0.0
Speed limit (20km/h)      0          0.0
Stop                     14          0.0

---

### IV. References

Clevert, Djork-Arne, et al. February 22, 2016. "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)". [https://arxiv.org/abs/1511.07289](https://arxiv.org/abs/1511.07289).

hengcherkeng. February 16, 2017. "My 99.37% solution to Udacity Nanodegree project P2". [https://medium.com/@hengcherkeng/this-is-a-test-bc767cfaf76e](https://medium.com/@hengcherkeng/this-is-a-test-bc767cfaf76e).

Mishkin, Dmytro et al. June 13, 2016. "Systematic evaluation of CNN advances on the ImageNet". [https://arxiv.org/abs/1606.02228](https://arxiv.org/abs/1606.02228).

Sermanet, Pierre and Yann LeCun. 2011. Traffic Sign Classification and Multi-Scale Convolutional Networks. [http://ieeexplore.ieee.org/document/6033589/](http://ieeexplore.ieee.org/document/6033589/).

Staravoitau, Alex. January 15, 2017. "Traffic signs classification with a convolutional network". [http://navoshta.com/traffic-signs-classification/](http://navoshta.com/traffic-signs-classification/)


