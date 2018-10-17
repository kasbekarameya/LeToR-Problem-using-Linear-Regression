# LeToR-Problem-using-Linear-Regression
One of the reasons why Search Engines today are the go-to place to find information is because of Search Engine Optimization. Any person searching on the internet would like their search results to be as accurate as possible. In order to make future searches possible, we can use Machine Learning algorithms like Linear Regression and Gradient Descent to enhance the Search Engine Optimization process. In this project, our goal is to use the concepts of Linear Regression and Gradient Decent algorithms to solve the problem of Learning to Rank in Information Retrieval on Microsoft’s LeToR Dataset. 

## Overview

Search Engine Optimization also abbreviated as SEO can be described as a process optimizing the results provided by a search algorithm in such a way that most relevant & frequent results are shown with greater importance.
Imagine that you have developed a website that provides great information about how to use farming tools. Hence you would like your website to be displayed prominently in the search results whenever someone searches for ‘farming tools’. In order to make such a task of selecting a website or link from multiple links, we use the concept of ranking. 
In this project, we use the concept of Learning to Rank (LeToR) or Machine-Learned Ranking(MLR), wherein we follow two approaches of implementing linear regression to rank links and web pages according to some predefined features.

### The LeToR Dataset
LeToR is a dataset specifically designed to research on Learning to the Rank problem. This dataset contains multiple pairs of input values & target values. The input values X can be described as values derived from the combination of the query as a row & document as a column. On the other hand, target values contain a total of three values i.e. 0,1, 2 where a large value like 2 will define that the query is perfectly matching with the document. 

To learn more about LeToR Dataset, visit this link: https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/

### Linear Regression Model
Linear Regression is one of the most used algorithms for prediction analysis. Linear regression can be described as a machine learning algorithm that is used to predict the value of the dependent variable Y, given enough amount of information about the independent variable Xv. More generally it uses the equation of the line to map values such that	
			
                                                        Y = W * X      
                                                        
Where Y is the dependent variable, X is the independent variable and W corresponds to the weights of the independent variables.   

## Algorithms & Equations used
The LeToR Dataset consists of 46 features that describe each query. But the problem is that even though these features are very helpful, they are too many to be mapped using a linear regression model. Hence we need to reduce them to a potential number, which can be done using clustering & basis functions. Using these concepts our linear regression equation will be modified to 

                                                    Y (x, w) = WT * Φ(X)
                                                    
Where w = (w0, w1, w2…, wn) is the weight vector of the training sample and Φ(x) = (Φ0, Φ1, Φ2, …., Φn  ) is a vector of M basis functions.

### K-Means Clustering Algorithm
K-means clustering is one of the most used clustering algorithms. It is a type of unsupervised learning algorithm that is used to group unlabeled data into definite categories. It works iteratively to add each data point into a specific category of data defined by the user. 
K-means clustering works on the fundamentals of mean or centroid that we have to define and iterate upon in the data. These centroids are random data points in the dataset placed at equal distances from each other. K-means algorithm uses these data points as a reference medium to cluster another similar type of data around them.
In this specific problem, we are using the data which is in the form of search queries and grouping them into clusters using K-means clustering. Hence a total of 69 thousand queries can be clustered together into 10 different query groups.  

### Radial Basis Functions
Radial Basis Function is a function that can be used to describe an element of the particular basis for a function space. Generally, Basis functions are used to represent the relationship between multiple non-linear inputs and target values in a dataset.
In this project, we are going to use Gaussian Radial Basis Functions. It calculates the radial basis functions using the following formula:
                                          Φj(x) = exp (−0.5 * (x – μj)T * ∑j-1 * (x – μj))
                                    
Where μj is the centroid obtained for the Basis function and ∑j-1 is the covariance matrix.

## Solving the LeToR Problem using Linear Regression
Now that we understand the concept of K-means Clustering & Radial Basis Functions, we can look into the two major approaches to finding out the weights of the input using linear regression on Learning to the Rank problem.

### Closed-Form Solution 
A solution to the machine learning problem can be defined as a closed form solution when we can use an equation or a group of equations to solve the given problem in terms of mathematical operations for a given dataset.
Another way of understanding closed form solution is to visualize a problem that can be solved using multiple methods. For example, in basketball, we can use different kinds of throws while throwing the ball into the basket. Hence now if we consider a method that can be represented in the form of an equation and can give one definite solution i.e. a basket in our example, then we can consider it as a closed form solution.  
Hence, in this project in order to find out the weights W of the input values we will be using the Moore- Penrose pseudo-inverse of a matrix Φ as follows:
                                                     W = (ΦT * Φ)-1 * ΦT * t
                                              
Where t is the target values vector and Φ is the design matrix containing radial basis functions Φj(xi)
Now using this equation, let’s try to reduce the error by varying the Hyper Parameters. Here we are evaluating the performance of the Linear Regression Model based on the following parameter;
    * Root Mean Square error (RMS): This parameter can be defined as the square root of the differences between the actual output and the expected output, which is wholly divided by the number of outputs.

#### Number of Basis Functions M:
Here we are using basis functions to map the nonlinear features of the dataset in a linear manner such that we can consider them at an input to a linear regression model. The method used to vary this parameter is a Grid Search approach wherein we have to manually try different values of the parameter to measure the variation in the output values.
As there are in total 46 features, it will be very difficult to provide them as an input to the Linear Regression model. Hence, using the Number of Basis Functions parameter (M), we can reduce these 46 features to 10 basis functions. 

#### Regularization Term 𝛌: 
In order to avoid overfitting of data in the linear regression model being used, we have to regularize the training process. Hence by including the regularization term like 𝛌 w,e will try to avoid overfitting of data and in turn, decrease the RMS error value.

Now based on the graph values obtained for both the Hyper Parameters above, we can determine that we have implemented a Closed Form Solution to the LeToR problem dataset. The graphs show that the output for all the RMS errors is constant around its best values. 
In order to understand why this happens, lets again refer to the closed form solution above. We know that, when we train a model to do the same task, we have to vary the Hyper Parameters so that it can work towards providing its best output value i.e. by updating the weights. 
On the other hand, as we have implemented and solved the problem using equations of the Closed Form Solution and not trained a model, all the values of the Hyper Parameters used are already set to their best values.
It is because of this fact that a Closed Form Solution to any kind of problem in machine learning will provide a best and finite output of the problem. Hence, now let’s try using the Stochastic Gradient Decent model to find the same minimum values of RMS errors that are obtained using the Closed Form Solution.

### Stochastic Gradient Decent Solution 
Gradient Descent is an optimization algorithm that uses gradients of the cost function to minimize the loss value. We use gradient descent to find the local minimum value of Weights W of the input values in this project.
One of the easiest ways of learning Gradient Decent is to take an example of the two mountains and a valley. Here we as a traveler have the goal of reaching the local minimum or lowest points of the valley. We can achieve these goals by taking steps along the side of the mountain to reach the goal. If we take too small steps, then it will take a lot of time to reach the goal and on the other hand, if we take to large steps then we may skip the goal altogether. 
Considering the example above we can now map the gradient descent algorithm into the following graph, where the two ends of the graph are the mountains and the value X is the lowest point of the valley. 

One of the most used variations of gradient descent algorithm is the Stochastic Gradient Descent Algorithm; also abbreviated as SGD. The term stochastic means that we compute a part of the problem representing the whole. Hence unlike the Batch Gradient Decent or Gradient Decent algorithm, in SGD we compute the gradient based on a single training sample as the stochastic approximation of the whole true gradient of the problem. 
Now using the Gradient Descent Algorithm, let’s try to reduce the error by varying the Hyper Parameters. Here we are evaluating the performance of the Linear Regression Model based on the following parameter;
    * Root Mean Square error (RMS): This parameter can be defined as the square root of the differences between the actual output and the expected output, which is wholly divided by the number of outputs.

#### Learning Rate 𝛈T
Learning Rate 𝛈T can be defined as the size of the steps taken by the gradient decent algorithm. With higher learning rate we can reach the goal faster, but there is a higher chance of divergence of the algorithm from the goal. On the other hand, if we define the step value as too small then the number of steps taken will increase and would lead to degradation of performance of the algorithm.

Now, as shown in the graph above when we try to select a large value or too small value for the learning rate 𝛈T, the accuracy of the linear regression algorithm drops and the RMS error increases significantly. From this observation, we can infer that as we increase the learning rate the step size of the gradient decent function increases. This may lead to the function missing the ultimate minimum point of the loss occurred. On the other hand, as we decrease the learning rate to a very small value, the number of steps increase as the step size decreases, leading to very little change in the loss value. After a number of variations, I can conclude that the optimal value for 𝛈T is between 0.01 and 0.1.

## Conclusion

This project aims at using machine learning to solve the problem of Learning to Rank (LeToR) in Information Retrieval. By performing this project, I was able to understand how Linear Regression can effectively fit on a nonlinear dataset i.e. dataset containing more than two features.  
Moreover, I was able to understand:
* How K-means clustering Algorithm works on unlabeled data
* How Radial Basis Functions can be used to fit non-linear data into curves.
* How we can use regularization to avoid overfitting of data
* How closed-form solution based on equations can be used to calculate weights
* How stochastic gradient descent algorithm can be used to calculate weights

In order to complete this project, the process used is as follows:
* Build a program representing the Closed Form Solution for the Linear Regression Model using Python
* Build a program representing the Stochastic Gradient Decent Solution for the Linear Regression Model using Python
* Vary the Hyper Parameters to obtain minimum possible Root Mean Square (RMS) error value for both the solutions.
In conclusion, I can state that by performing this project I was able to understand how to use a linear regression model to solve the problem of mapping and ranking documents based on search queries.
