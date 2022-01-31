# imports

from sklearn import datasets
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class Generator:

    def __init__ (self):
        self.columns = ['x', 'y', 'labels']

    def makeDF(self, number_of_points):
        return pd.DataFrame(self.makeData(number_of_points), columns=self.columns)
        
    def makeData(self, number_of_points):
        _x, _y = datasets.make_blobs(
            n_samples = number_of_points,
            n_features = 2,
            centers = 2,
            cluster_std = 1.05,
            random_state = random.randint(1,100))
        return list(zip(_x[:,0], _x[:,1], _y))

class Plotter:

    def samples_by_category(self, df):
        return sns.scatterplot(
            x = df.columns[0], 
            y = df.columns[1], 
            data = df,
            hue = df.columns[2])

    def samples_by_category_with_decision_boundary(self, df, theta):
        _p = self.samples_by_category(df)
        plt.plot([-12, -4], [2, 4], linewidth=2)

class RandomData: 

    def __init__(self, number_of_points):
        self.df = Generator().makeDF(number_of_points)
        self.plotter = Plotter()

    def plot(self):
        self.plotter.samples_by_category(self.df)

class Perceptron:

    def __init__(self, data):
        self.epochs = 100
        self.learning_rate = 0.1
        self.data = data
        self.labels = self.data.iloc[:,-1]
        self.samples = self.data.iloc[:,:-1]
        self.sample_count = self.samples.shape[0]
        self.feature_count  = self.samples.shape[1]
        self.theta = np.zeros((self.feature_count + 1, 1))
        self.misclassified_counts = [0 for _ in range(self.sample_count)]
        self.activation_function = ActivationFunction(0.0)
        self.plotter = Plotter()

    def train(self):
        for _ in range(self.epochs):
            for _i, _sample_i in enumerate(self.samples.iterrows()):
                print(self.samples.iloc[_i,:])
                self.misclassified_counts[_i] = 0

    def neuron(self, index, sample):
        _prediction = self.predict(sample)
        self.evaluate_prediction(index, sample, _prediction)
    
    def predict(self, sample):
        _processed_sample = np.insert(sample, 0, 1).reshape(-1,1)
        _dot_product = np.dot(_processed_sample.T, _theta)
        _prediction = self.activation_function.check(_dot_product)
        return _prediction
    
    def evaluate_prediction(self, index, sample, prediction):
        if np.squeeze(prediction, self.labels[index]) == 0:
            return
        self.update_theta(index, sample, prediction)
        self.increment_misclassification_count(index)

    def update_theta(self, index, sample, prediction):
        self.theta += self.learning_rate*sample*(self.labels(index) - prediction)

    def increment_misclassification_count(self, index):
        self.misclassified_count[index] += 1

    def plot_boundary(self):
        self.plotter.samples_by_category_with_decision_boundary(self.data, self.theta)

    def print_initialization(self):
        print(f'epochs: {self.epochs}')
        print(f'learning rate: {self.learning_rate}')
        print(f'sample count: {self.sample_count}')
        print(f'feature count: {self.feature_count}\n')
        print(f'samples:\n{self.samples.head()}\n')
        print(f'labels:\n{self.labels.head()}\n')
        print(f'theta:\n{self.theta}\n')

class ActivationFunction:

    def __init__(self, threashold):
        self.threashold = threashold

    def check(self, data):
        return [activate(_x) if _x > self.threashold else deactivate(_x) for _x in data]

    def activate(self, data):
        return 1.0

    def deactivate(self, data):
        return 0.0

_data = RandomData(555)
_perceptron = Perceptron(_data.df)
_perceptron.epochs = 10
_perceptron.train()
_perceptron.plot_boundary()