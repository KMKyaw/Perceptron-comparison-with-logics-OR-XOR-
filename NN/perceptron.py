import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        # initialize the weight matrix and store the learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
        
    def step(self, x):
        # apply the step function
        return 1 if x > 0 else 0


    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]


		# loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
		# loop over each individual data point
        	for (x, target) in zip(X, y):
                    p = self.step(np.dot(x, self.W))
                    if p != target:
                        error= p - target            
                        self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
		# ensure our input is a matrix
        X =np.atleast_2d(X)
		# check to see if the bias column should be added
        if addBias:
		# insert a column of 1’s as the last entry in the feature
		# matrix (bias)
            X= np.c_[X, np.ones((X.shape[0]))]

		# take the dot product between the input features and the
		# weight matrix, then pass the value through the step
		# function
        return self.step(np.dot(X, self.W))	