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
        if theta[2] == 0: return
        _p = self.samples_by_category(df)
        _feature_1_range = [df.x.min(), df.x.max()]
        _slope = -theta[1]/theta[2]
        _intercept = -theta[0]/[theta[2]]
        _feature_2_range = _slope * _feature_1_range + _intercept
        plt.plot(_feature_1_range, _feature_2_range, linewidth=2)

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
                self.neuron(_i, np.array([_sample_i[1]['x'], _sample_i[1]['y']]))
                self.misclassified_counts[_i] = 0

    def neuron(self, index, sample):
        _processed_sample = np.insert(sample, 0, 1).reshape(-1,1)
        _prediction = self.predict(_processed_sample)
        self.evaluate_prediction(index, _processed_sample, _prediction)
    
    def predict(self, sample):
        _dot_product = np.dot(sample.T, self.theta)
        _prediction = self.activation_function.check(_dot_product)
        return _prediction
    
    def evaluate_prediction(self, index, sample, prediction):
        if np.squeeze(prediction) - self.labels[index] == 0:
            return
        self.update_theta(index, sample, prediction)
        self.increment_misclassification_count(index)

    def update_theta(self, index, sample, prediction):
        self.theta += self.learning_rate*sample*(self.labels[index] - np.squeeze(prediction))

    def increment_misclassification_count(self, index):
        self.misclassified_counts[index] += 1

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
        return [self.activate(_x) if _x > self.threashold else self.deactivate(_x) for _x in data]

    def activate(self, data):
        return 1.0

    def deactivate(self, data):
        return 0.0


_data = RandomData(555)
_perceptron = Perceptron(_data.df)
_perceptron.epochs = 10
_perceptron.train()
print(_perceptron.theta)
_perceptron.plot_boundary()