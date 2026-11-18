# Rust Machine Learning Examples
A few Rust examples from studying the [Machine Learning Fundamentals](https://cseweb.ucsd.edu/~dasgupta/151/schedule.html) course taught by Sanjoy Dasgupta (UCSD). Not homework, just a personal project. AI was utilized in making these. 

## 1-nn
A 1 nearest neighbor implementation on the [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). Using Euclidean distance, find the closest training image to classify a new image from the test set. Error rate is 3.09% as shown in the course.

## k-nn
A k nearest neighbor implementation on the [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). Still using Euclidean distance, find the closest of k nearest neighbors to classify an new image from the test set. In the Rust code k is set to 3, but can be changed. It should be noted that an odd numbered k is desirable so that you can choose an image among the others. The error rate I got was 2.95% with k=3 while Professor Dasgupta got 2.94%. 

## Spam-bayes
I thought it would be fun to use the [ENRON dataset](https://github.com/MWiechmann/enron_spam_data) and Bayes theorem to predict whether an email is spam or legitimate (ham). This is done by first calculating the prior probabilities then the word probabilities; apply Bayes's theorem by combining the prior and word probabilities, and finally making a prediction based on the result.

<img width="589" height="160" alt="2" src="https://github.com/user-attachments/assets/73ab5380-5662-4d9f-8793-24bc169734fd" />


## Spam-perceptron
In this one we use the [ENRON dataset](https://github.com/MWiechmann/enron_spam_data) and build a Perceptron to classify a new email as spam or legitimate (ham). The perceptron learns which words are associated with spam and which are associated with legitimate emails. It does this by building a vocabulary, initializing weights and biases, making predictions, adjusting from errors, and repeating the process over 10  epochs (one epoch is one pass through the dataset).

## Linear Regression Visualization
This visualization was inspired by Nikita Lisitsa's work that can be found [here](https://lisyarus.github.io/webgl/classification.html). The idea was to only have it show linear regression instead of logistic regression. Here is what it should look like with some example points and the generated linear regression line:  

<img width="585" height="438" alt="1" src="https://github.com/user-attachments/assets/a2c96fcb-836a-4a4f-a687-321892cbb3ba" />

## Winery Gaussian Model
In this one we recreate the Professor's explanation of the [Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine). Since we don't know the exact split used by Professor Dasgupta, we will just take the non-randomized first n wines for the training and test sets. We know the training set should add up to 130 and the test set should add up to 48. This yielded results similar to the Professor's. As can be seen, mine is on the left, and the professor's is on the right. The Winery 1 Alcohol Statistics I got were a Mean μ of 13.74 and Variance σ² of 0.22.

<img width="900" height="580" alt="wines" src="https://github.com/user-attachments/assets/45025e92-ae11-43d0-95bc-b0c7b01a9cb9" />


## Logistic Regression Visualization
#todo

## Covariance Matrix
#todo

### To be noted:
Regarding the MNIST dataset, the Kaggle links above are what I used in my project. However the author of the MNIST dataset, Yann Lecun also has it available [here](https://huggingface.co/datasets/ylecun/mnist). 
