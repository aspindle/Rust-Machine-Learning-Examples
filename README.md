# Rust-Machine-Learning-Examples
A few Rust examples from studying the Machine Learning Fundamentals course taught by Sanjoy Dasgupta (UCSD). Not homework, just a personal project. AI was utilized in making these. 

## 1-nn
A 1 nearest neighbor implementation on the [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). Using Euclidean distance, find the closest training image to classify a new image from the test set. Error rate is 3.09% as shown in the course.

## k-nn
A k nearest neighbor implementation on the [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). Still using Euclidean distance, find the closest of k nearest neighbors to classify an new image from the test set. In the Rust code k is set to 3, but can be changed. It should be noted that an odd numbered k is desirable so that you can choose an image among the others. The error rate I got was 2.95% with k=3 while Professor Dasgupta got 2.94%. 

## Spam-bayes
I thought it would be fun to use the [ENRON dataset](https://github.com/MWiechmann/enron_spam_data) and Bayes theorem to predict whether an email is spam or legitimate (ham). This is done by first calculating the prior probabilities then the word probabilities; apply Bayes's theorem by combining the prior and word probabilities, and finally making a prediction based on the result.

## Linear Regression Visualization
This visualization was inspired by Nikita Lisitsa's work that can be found [here](https://lisyarus.github.io/webgl/classification.html). The idea was to only have it show linear regression instead of logistic regression. The plan is to remake the logistic regression in Rust as well. #todo



### To be noted:
Regarding the MNIST dataset, the Kaggle links above are what I used in my project. However the author of the MNIST dataset, Yann Lecun also has it available [here](https://huggingface.co/datasets/ylecun/mnist) despite them having some not-so-great political opinions. 
