# Haar-Cascades-
Implementing Viola Jones Algorithm, popularly knows as Haar Cascades in Python

VIOLA JONES OBJECT DETECTION ALGORITHM 
(HAAR CASCADES)

1.	ABSTRACT
Haar cascade popularly known as Viola Jones object detection algorithm was developed by Paul Viola and Michael Jones in 2001. Haar cascade is primarily used in face detection. The algorithm has four stages Haar Feature Selection, Creating an Integral Image, Adaboost Training and Cascading Classifiers. The algorithm looks for most relevant features such as eyes, nose, lips and eyebrows. 
Haar features are very similar to convolution kernels and are most relevant and powerful features for face detection. Haar features are basically of two types edge features and line features. Eyebrows and eyes are detected by edge features and lips are usually detected by line features. Viola Jones algorithm will compare how close the real scenario is to an ideal Haar kernel. The closer the value to one the more likely a Haar like feature is detected.
A real image may contain hundreds of pixels and computation for detection of Haar features is done several times. The time complexity of these operations is O(N2). Integral image reduces the quadratic time complexity to linear constant time complexity and thus speeds up the entire process of face detection. Adaboost classifier is a boosting algorithm which combines outcomes of weak learners to make a more robust model whereas cascades help in elimination of non-facial features from essential facial features and thus reduce the total number of features.
Apart from face detection Viola Jones algorithm is extensively used in obstacle target detection and license number plate detection. Haar cascade detection is one of the oldest yet powerful algorithms invented. It has been there since long before deep learning became famous. They can be easily accessed and trained with OpenCV methods in Python. 
The seminar will begin with an introduction to Haar Cascade and problems associated with face or object detection. Each and every step of the algorithm will have a detailed explained and enough examples will be given for better understanding. High level description of concepts like Haar Wavelet Transformation and Boosting will also be covered. The seminar will also deal with practical applications of Haar Cascades and how it is implemented in Python 3.0.

2.	HISTORY OF VIOLA JONES OBJECT DETECTION ALGORITHM
Object detection can be defined as the ability to identify or detect objects like trees, human faces, cars, etc in an image or a video. It is an application of computer vision. The primary aim of face detection is to determine whether there is any face in an image or not. This is done by finding facial features like eyes, nose and lips. Viola Jones algorithm popularly known as Haar Cascades was developed by Paul Viola and Micheal Jones in 2001. Though the algorithm can be used for various object detection problems it is primarily used for face detection. Viola Jones algorithm is fast, powerful and robust.
Since the last decade face detection has taken pace. Public concern for security, the need for identity verification in the fast growing digital world, and face mask detection during the pandemic are some of the reasons which have made face detection algorithms very popular recently. Face detection becomes a challenge where the background, head pose, and illumination are varying.

3.	DIFFERENCE BETWEEN HAAR CASCADES AND CNN
Convolution Neural Networks (CNN) are powerful tools used for image classification and segmentation. The major difference between Haar Cascades and CNN is the kernel that convolves with the input image [1]. In Haar Cascades the values of the kernel are determined manually whereas in a CNN the values of the filter (kernel) are determined during training of the network. Haar Cascades perform well when edge and line features are clear and may fail to detect partially covered faces. A convolution kernel, on the other hand, can recognize partially covered faces with high accuracy (depending on the quality of the training data). Large image dataset is needed in a CNN and hence lot of training is required whereas Haar features are not required to be train and a good classifier can be created using a relatively small dataset. The last point makes Haar features computationally less expensive compared to CNNs.





4.	STEPS INVOLVED IN VIOLA JONES FACE DETECTION ALGORITHM
 
Fig 1: Flowchart showing steps involved in Viola Jones Face Detection Algorithm


4.1	SELECTION OF HAAR-LIKE-FEATURES
The first step of Viola Jones Face Detection Algorithm is selection of Haar-like-features from the input image. These features on the images make it easy to find out the edges or the lines in the image, or to pick up areas where there is a sudden change in the intensities of the pixels [2]. Haar features can easily detect eyes, nose, lips, eyebrows, and forehead. These are the most relevant features as far as the human face is considered. They are very powerful for face detection. Haar features works on the concept of Haar wavelets, proposed by Hungarian mathematician Alfred Haar in 1909. Haar wavelets are a sequence of rescaled square shaped functions and are very similar to Fourier Analysis.
There are two types of Haar features (1) Line features and (2) Edge features. Eyebrows and eyes have darker pixels compared to forehead and are easily detected by edge features. Lips are usually detected by line features. Typical line and edge Haar features are shown below.

                                                                           
Fig 2: Line Features                                           Fig 3: Edge Features

In an ideal Haar feature zero corresponds to black (dark) pixels and one corresponds to white (bright) pixels. A typical gray scale image has pixel values ranging from 0 to 255 and values closer to 0 corresponds to darker pixels whereas values closer to 255 corresponds to brighter pixels. Haar cascades will compare how close the real scenario is to the ideal case. This is done by taking the sum of all the pixel values lying in the darker area of the Haar feature and sum of the pixel values lying in the lighter area of the Haar feature. And then computing their difference. For an ideal Haar feature the difference should be one whereas for a Haar like feature the difference must be close to one. The closer the value to one, the more likely we have found a Haar like feature.

         
Fig 4: Ideal Haar Feature

4.2	COMPUTING INTEGRAL IMAGE
Haar features need to transverse from top left of the input image to the bottom right horizontally as well as vertically to search for a particular feature [2]. Also all possible sizes of the haar features will be applied. Haar features shown in Figure 2 above are responsible for finding out if there is a darker region surrounded by a lighter region on either side or vice versa. Haar features shown in the Figure 3 above are responsible for finding out edges in a horizontal or a vertical direction. The last feature is Figure 3 finds out change in pixel intensities across diagonals. Since Haar features of different sizes traverse over the entire image we need to compute the average of a given region several times. An image usually contains hundreds of pixel values and thus the entire operation could be hectic even for a high performance machine.The time complexity of these operations is O(N2). 
To address the above issue, the concept of integral image is used to perform the operation. Each pixel of the integral image (see Figure 6) is calculated by taking the sum of all the pixels lying to its left and above in the original image (see Figure 5). This is helpful in the sense that if we want to calculate the sum of pixel values in a region we don’t have to consider all the pixels in the original image. We just need to consider four constant values from the integral image each time for each time for any feature size. This reduces the quadratic time complexity to linear constant time complexity.
                                                  
                          Fig 5: Original Image                                                       Fig 6: Inegral Image
                   
4.3	ADABOOST TRAINING
Haar features can find more than 160,000 features and most of the features are irrelevant i.e most of them are not facial features. There is a need for a feature selection technique which will not only select a subset of features performing better than others, but also will eliminate the irrelevant ones. This is done by Adaboost algorithm which selects most relevant features and trains them with cascading classifiers. Adaboost is an example of a boosting algorithm which builds a strong classifier by combining simple weak learners which perform better than only a random guess. Adaboost finds a weighted linear combination of the outputs of the weak learners and generates an output by giving more weight to the weak learners which perform badly (more misclassifications) and less weight to the correctly classified items. With this technique, the final set of features can get reduced to a total of 6000 features.

 
                     Fig 7: Visual representation of Adaboost
4.4	CASCADE FILTERING
Lot of processing is required even with 6000 features [2]. To reduce the computation time researchers have proposed a method known as cascade classifiers. A simple idea behind cascade classifiers is that all Haar features need not to traverse on each and every window. If there are no Haar-like-features in a particular window then we can say that facial features are not present there. In this case we can move to the next window and check for facial features. If the initial stages of cascade filtering  won’t detect anything on the window then that window is completely discarded from the remaining process. This saves a lot of processing time as the irrelevant features are not selected in the majority of stages. The second stage of processing gets triggered only when the features are detected in the first stage and the process continues. Finally if we pass through all the stages and nothing fails, we detect a face. Cascade filtering reduces the number of false negatives by removing windows having no facial features.

5.	CONCLUSION
Haar Cascade detection is one of the oldest and powerful face detection algorithms invented. It has been there since long before deep learning famous. Haar kernel selects relevant facial features from the image based on cascade filtering. Integral images reduces computation time whereas Adaboost helps in making the classifier robust. andEase of understanding also makes Viola Jones algorithm pioneer in face detection. Haar Cascade can be easily implemented in Python 3.0 [3]. One has to make sure all latest version of Python, mathplotlib, and OpenCV are installed on your machine. The haar cascade can be downloaded from Github repository [4].

6.	REFERENCES
[1] Chi Feng Wang. What’s The Difference Between Haar-Feature Classifiers and Convolution Neural Networks? August 4, 2008. https://towardsdatascience.com/whats-the-difference-between-haar-feature-classifiers-and-convolutional-neural-networks-ce6828343aeb
[2] Girija Shankar Behera. Face Detection with Haar Cascade. December 24, 2020. https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08
[3] Aashal Kamdar. Haar Cascade for Object Detection. 06 May, 2021 https://www.geeksforgeeks.org/python-haar-cascades-for-object-detection/
[4] OpenCV. https://github.com/opencv/opencv/tree/master/data/haarcascades

