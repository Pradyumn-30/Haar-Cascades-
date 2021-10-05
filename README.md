# Haar-Cascades
Implementing Viola Jones Algorithm, popularly knows as Haar Cascades in Python

VIOLA JONES OBJECT DETECTION ALGORITHM 
(HAAR CASCADES)

Haar cascade popularly known as Viola Jones object detection algorithm was developed by Paul Viola and Michael Jones in 2001. Haar cascade is primarily used in face detection. The algorithm has four stages Haar Feature Selection, Creating an Integral Image, Adaboost Training and Cascading Classifiers. The algorithm looks for most relevant features such as eyes, nose, lips and eyebrows. 
Haar features are very similar to convolution kernels and are most relevant and powerful features for face detection. Haar features are basically of two types edge features and line features. Eyebrows and eyes are detected by edge features and lips are usually detected by line features. Viola Jones algorithm will compare how close the real scenario is to an ideal Haar kernel. The closer the value to one the more likely a Haar like feature is detected.
A real image may contain hundreds of pixels and computation for detection of Haar features is done several times. The time complexity of these operations is O(N2). Integral image reduces the quadratic time complexity to linear constant time complexity and thus speeds up the entire process of face detection. Adaboost classifier is a boosting algorithm which combines outcomes of weak learners to make a more robust model whereas cascades help in elimination of non-facial features from essential facial features and thus reduce the total number of features.
Apart from face detection Viola Jones algorithm is extensively used in obstacle target detection and license number plate detection. Haar cascade detection is one of the oldest yet powerful algorithms invented. It has been there since long before deep learning became famous. They can be easily accessed and trained with OpenCV methods in Python.
