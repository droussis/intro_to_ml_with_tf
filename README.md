# Introduction to Machine Learning with TensorFlow

In this repository, you will find the projects that I carried out for the __"Introduction to Machine Learning with TensorFlow"__ Udacity Nanodegree program.

Each folder contains a project, its Anaconda environment and its deliverables. Note that the data that are used, as well as some necessary functions are missing and thus, the notebooks cannot be reproduced. However, feel free to to use and experiment with the application of Project 2 by following the guidelines below.

---

## Project 1 - Finding Donors for CharityML

Python version: 3.5.5

In this project, the student has to implement several supervised learning algorithms and answer certain questions related to the task of classifying individuals to two classes; those that earn more than $50,000 and those that do not. This would help CharityML, a fictitious charity organization, determine where to send letters asking for donations, given that the previous experience has shown that only individuals with an annual income larger than $50,000 actually make donations. I personally implement and evaluate three classifiers: 

- Logistic Regression
- Decision Trees
- Gradient Boosting 

Moreover, the Gradient Boosting classifier which proved to have the best performance (in terms of F-score and accuracy) is fine-tuned using a grid search of some of its hyper-parameters. Additional information can be found in the Jupyter notebook.

---

## Project 2 - Create Your Own Image Classifier - TensorFlow

Python version: 3.7.6

This project is divided into two parts. 

### Part 1 - Development Notebook
In the first part, the student has to load the __Oxford Flowers 102__ dataset (contained in TensorFlow Datasets), do some minimal exploration, normalization and preprocessing and afterwards, load a pre-trained network (MobileNet) which is used to train a new neural network which can classify the flower images to their corresponding classes.

### Part 2 - Command Line Application
In the second part, the student has to implement a command line application which is given an image, a saved pre-trained model and some optional parameters and returns the predicted classes. Our model managed to score 78.73% accuracy on the test set (previously unseen data).

In particular, in order to use the command line application, you have to follow these steps:
1. Clone the repository and move to the 'image_classifier' directory
2. Create a conda environment with all the requirements using: 
   ```
   $ conda env create -f environment.yml
   ```
3. Use the ```predict.py``` to predict the label or class of a given flower image: 
   ```
   $ python predict.py saved_model --top_k k --category_names map.json 
   ```

There are also two optional parameters which can be used:
* ```--top_k```: Allows the user to print out the top k most likely classes that the image belongs to along with the associated probabilities.
* ```--category_names```: Allows the user to specify the path to a JSON file that maps the labels to the category names.

* __Example Usage__:

```$ python predict.py test_images/wild_pansy.jpg flower_classifier.h5 --category_names label_map.json --top_k 5```

