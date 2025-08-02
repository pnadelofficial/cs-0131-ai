import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import tqdm
    tqdm_here = True
except ModuleNotFoundError:
    tqdm_here = False

np.random.seed(1997)

def make_one_hot(labels):
    num = int(np.max(labels) + 1)
    one_hot = np.zeros((labels.size, num))
    one_hot[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot

class DataLoader:
    def __init__(self, data_path="data.txt"):
        self.data_path = data_path

        with open(self.data_path, "r") as f:
            lines = f.readlines()
        data = [line.strip().split(",") for line in lines][:-1] 
        df = pd.DataFrame(data, columns=["s_len", "s_wid", "p_len", "p_wid", "class"])
        label_mapping = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2
        }
        df['class'] = df['class'].map(label_mapping)
        df = df.astype(float)
        self.df = df
    
        self.normalize_data()
        self.train_test_valid_split()

    def normalize_data(self):
        targets = self.df[['class']]
        normed = self.df[["s_len", "s_wid", "p_len", "p_wid"]].apply(lambda x: (x - x.min())/(x.max() - x.min()))
        self.df = pd.concat([normed, targets], axis=1)

    def train_test_valid_split(self, train_ratio=0.6, test_ratio=0.5):
        train_mask = np.random.rand(len(self.df)) < train_ratio

        self.train = self.df[train_mask]
        test_valid = self.df[~train_mask]

        test_mask = np.random.rand(len(test_valid)) < test_ratio
        self.test = test_valid[test_mask]
        self.valid = test_valid[~test_mask]
        
        print(f"Data splits: Training length: {len(self.train)}; Testing length: {len(self.test)}; Valid length: {len(self.valid)}")

class Sigmoid:
    def __init__(self):
        pass
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def backward(self, x):
        return x * (1.0 - x)

class CrossEntropyLoss:
    def __init__(self):
        pass
    def __call__(self, y, y_pred):
        return np.mean(-np.sum(y * np.log(y_pred), axis=-1))
    def backward(self, y, y_pred):
        return y - y_pred

class NeuralNetwork:
    def __init__(self, dataloader, hidden_layers, hidden_dim):
        self.dataloader = dataloader
        self.train = self.dataloader.train
        self.test = self.dataloader.test
        self.valid = self.dataloader.valid
        self.input_layer_size = self.train.iloc[:, :-1].shape[-1]
        self.output_layer_size = int(np.max(self.train.iloc[:, -1]) + 1)

        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

        self.model = {}
        self.model["W_0"] = None
        self.model["b_0"] = None
        self.model[f"W_{self.hidden_layers+1}"] = None
        self.model[f"b_{self.hidden_layers+1}"] = None
        
    def initialize(self, activation=Sigmoid):
        self.model["W_0"] = np.random.randn(self.input_layer_size, self.hidden_dim) - 0.5
        self.model["b_0"] = np.zeros((1, self.hidden_dim))
        self.model["act_0"] = activation()

        for i in range(1, self.hidden_layers+1):
            self.model[f"W_{i}"] = np.random.randn(self.hidden_dim, self.hidden_dim) - 0.5
            self.model[f"b_{i}"] = np.zeros((1, self.hidden_dim))
            self.model[f"act_{i}"] = activation()
        
        self.model[f"W_{self.hidden_layers+1}"] = np.random.randn(self.hidden_dim, self.output_layer_size) - 0.5
        self.model[f"b_{self.hidden_layers+1}"] = np.zeros((1, self.output_layer_size))
        self.model[f"act_{self.hidden_layers+1}"] = activation()
    
    def __call__(self, X, y, loss_fn=None, valid=False):
        z = None
        for i in range(self.hidden_layers+2):
            weight = self.model[f"W_{i}"]
            bias = self.model[f"b_{i}"]
            activation = self.model[f"act_{i}"]
            if z is None:
                z = X
            z = activation(np.dot(z, weight) + bias)
            if not valid:
                self.model[f"z_{i}"] = z
    
        if loss_fn is None:
            return (y == z.argmax(axis=1)).sum() / len(y)
        else:
            loss = loss_fn()(make_one_hot(y), z)
            return loss

    def backward(self, X, y, loss_fn=CrossEntropyLoss):
        current_layer = self.hidden_layers + 1
        z_current = self.model[f"z_{current_layer}"]
        dz = loss_fn().backward(make_one_hot(y), z_current)

        self.model[f"db_{current_layer}"] = dz.sum(axis=0, keepdims=True)
        dz *= self.model[f"act_{current_layer}"].backward(z_current)

        if current_layer == 0:
            previous_activation = X
        else:
            previous_activation = self.model[f"z_{current_layer-1}"]
        self.model[f"dW_{current_layer}"] = np.dot(previous_activation.T, dz)
        
        for layer in range(current_layer-1, -1, -1):
            next_layer = layer + 1
            dz = (np.dot(dz, self.model[f"W_{next_layer}"].T)) * self.model[f"act_{layer}"].backward(self.model[f"z_{layer}"])

            self.model[f"db_{layer}"] = dz.sum(axis=0, keepdims=True)
            if layer == 0:
                previous_activation = X
            else:
                previous_activation = self.model[f"z_{layer-1}"]
            self.model[f"dW_{layer}"] = np.dot(previous_activation.T, dz)
    
    def update(self, lr):
        for layer in range(self.hidden_layers + 2):
            self.model[f"W_{layer}"] += lr * self.model[f"dW_{layer}"]
            self.model[f"b_{layer}"] += lr * self.model[f"db_{layer}"]

        