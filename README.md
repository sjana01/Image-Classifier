# Image-Classifier

This is an image classification project. In this project we use Convolutional Neural Networks (CNN) to classify images. The CNN architecture we are using is one of 'VGG19' or 'ResNet50'. The training dataset for these CNN models have 1000 different classes of images. For more information on these models check the following websites: 

https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19

https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50

We also build a Flask app, which can take any image file as input and predict what that image is about. This app only works for simple images as the model is only trained on a dataset with 1000 labels. A Flask app is a web application created using the Flask web framework, which is written in Python. Flask is a micro web framework that provides a simple way to build web applications. It includes routing and handling of HTTP requests, and a template engine for rendering views. These Flask apps can be built and run on a local development server.

To run this Flask app, run the command "python app.py" in a terminal or command prompt. The app will run on a local development server and be accessible at "http://localhost:5000" in your web browser.
