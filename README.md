# Object-Detection
This project can detected Object from Video using Tensorflow Object detection API.

Object Detection
Creating accurate Machine Learning Models which are capable of identifying and localizing multiple objects in a single image remained a core challenge in computer vision. But, with recent advancements in Deep Learning, Object Detection applications are easier to develop than ever before. TensorFlow’s Object Detection API is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models.
So , In this Object Detection Tutorial, I’ll be covering the following topics:
o	What is Object Detection?
o	Different Applications of Object Detection
o	Object Detection Workflow
o	What is Tensor flow?
o	Object Detection with Tensorflow (Demo)
o	Real-Time/Live Object Detection (Demo)
You can go through this real-time object detection Tutorial, where our Deep Learning Training expert is discussing how to detect an object in real-time using TensorFlow. Real-Time Object Detection with TensorFlow 
This will provide you with a detailed and comprehensive knowledge of TensorFlow Object detection and how it works. It will also provide you with the details on how to use Tensorflow to detect objects in the deep learning methods.

What is Object Detection?
Object Detection is the process of finding real-world object instances like car, bike, TV, flowers, and humans in still images or Videos. It allows for the recognition, localization, and detection of multiple objects within an image which provides us with a much better understanding of an image as a whole. It is commonly used in applications such as image retrieval, security, surveillance, and advanced driver assistance systems (ADAS).
Object Detection can be done via multiple ways:
•	Feature-Based Object Detection
•	Viola Jones Object Detection
•	SVM Classifications with HOG Features
•	Deep Learning Object Detection
In this Object Detection Tutorial, we’ll focus on Deep Learning Object Detection as Tensorflow uses Deep Learning for computation.
 
Let’s move forward with our Object Detection Tutorial and understand it’s various applications in the industry.
Applications Of Object Detection
Facial Recognition:
A deep learning facial recognition system called the “DeepFace” has been developed by a group of researchers in the Facebook, which identifies human faces in a digital image very effectively. Google uses its own facial recognition system in Google Photos, which automatically segregates all the photos based on the person in the image. There are various components involved in Facial Recognition like the eyes, nose, mouth and the eyebrows. 
People Counting:
Object detection can be also used for people counting, it is used for analyzing store performance or crowd statistics during festivals. These tend to be more difficult as people move out of the frame quickly.
It is a very important application, as during crowd gathering this feature can be used for multiple purposes.
 Industrial Quality Check:
Object detection is also used in industrial processes to identify products. Finding a specific object through visual inspection is a basic task that is involved in multiple industrial processes like sorting, inventory management, machining, quality management, packaging etc. Inventory management can be very tricky as items are hard to track in real time. Automatic object counting and localization allows improving inventory accuracy.
Self-Driving Cars:
Self-driving cars are the Future, there’s no doubt in that. But the working behind it is very tricky as it combines a variety of techniques to perceive their surroundings, including radar, laser light, GPS, odometry, and computer vision.
Advanced control systems interpret sensory information to identify appropriate navigation paths, as well as obstacles and once the image sensor detects any sign of a living being in its path, it automatically stops. This happens at a very fast rate and is a big step towards Driverless Cars.
Security:
Object Detection plays a very important role in Security. Be it face ID of Apple or the retina scan used in all the sci-fi movies.
It is also used by the government to access the security feed and match it with their existing database to find any criminals or to detect the robbers’ vehicle. The applications are limitless.
Object Detection Workflow
Every Object Detection Algorithm has a different way of working, but they all work on the same principle.
Feature Extraction: They extract features from the input images at hands and use these features to determine the class of the image. Be it through MatLab, Open CV, Viola Jones or Deep Learning.
Now that you have understood the basic workflow of Object Detection, let’s move ahead in Object Detection Tutorial and understand what Tensorflow is and what are its components?
What is TensorFlow?
Tensorflow is Google’s Open Source Machine Learning Framework for dataflow programming across a range of tasks. Nodes in the graph represent mathematical operations, while the graph edges represent the multi-dimensional data arrays (tensors) communicated between them.

Tensors are just multidimensional arrays, an extension of 2-dimensional tables to data with a higher dimension. There are many features of Tensorflow which makes it appropriate for Deep Learning. So, let’s see how we can implement Object Detection using Tensorflow.
