import numpy as np
import pickle
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class ActivationFunction:
    @staticmethod
    def relu(y_feed):
        return np.maximum(0, y_feed)
    
    @staticmethod
    def relu_derivative(y_feed):
        return np.where(y_feed > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(y_feed):
        return np.where(y_feed > 0, y_feed, y_feed * 0.01)
    
    @staticmethod
    def leaky_relu_derivative(y_feed):
        return np.where(y_feed > 0, 1, 0.01)
    
    @staticmethod
    def softmax(y_feed):
        exps = np.exp(y_feed - np.max(y_feed, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def softmax_derivative(self, y_feed):
        return ActivationFunction.softmax(y_feed) * (1 - ActivationFunction.softmax(y_feed))

class WeightsInitialization:
    @staticmethod
    def xavier_initialization(n_input, n_output): # Bisa buat Sigmoid/Tanh
        return np.random.randn(n_input, n_output) * np.sqrt(1 / n_input)
    
    @staticmethod
    def he_initialization(n_input, n_output): # Bisa buat Leaky ReLU/ReLU/Softmax
        return np.random.randn(n_input, n_output) * np.sqrt(2 / n_input)
    
    @staticmethod
    def random_initialization(n_input, n_output):
        return np.random.randn(n_input, n_output) 

class FaceRecognitionModel:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.learning_rate = 0.01
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.weights_gradients = []
        self.biases_gradients = []
        
        self.initialize_parameters()
    
    def initialize_weights(self):
        input_size = self.input_size
        for size in self.hidden_layer_sizes + [self.output_size]:
            self.weights.append(WeightsInitialization.he_initialization(input_size, size))
            self.weights_gradients.append(np.zeros((input_size, size)))
            input_size = size

    def initialize_biases(self):
        for size in self.hidden_layer_sizes + [self.output_size]:
            self.biases.append(np.zeros(size))
            self.biases_gradients.append(np.zeros(size))
    
    def initialize_activations(self):
        for i in range(len(self.hidden_layer_sizes)):
            self.activations.append(ActivationFunction.leaky_relu)
        self.activations.append(ActivationFunction.softmax)
    
    def initialize_parameters(self):
        self.initialize_weights()
        self.initialize_biases()
        self.initialize_activations()
    
    def forward_propagation(self, X):
        self.z = []
        self.a = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = self.activations[i](z)
            self.a.append(a)
        
        return self.a[-1]
    
    def backward_propagation(self, X, y, y_pred):
        m = X.shape[0]
        self.weights_gradients = [np.zeros_like(w) for w in self.weights]
        self.biases_gradients = [np.zeros_like(b) for b in self.biases]
        
        dz = y_pred - y
        for i in reversed(range(len(self.weights))):
            self.weights_gradients[i] = np.dot(self.a[i].T, dz) / m
            self.biases_gradients[i] = np.sum(dz, axis=0) / m
            if i > 0:
                if self.activations[i-1] == ActivationFunction.leaky_relu:
                    dz = np.dot(dz, self.weights[i].T) * ActivationFunction.leaky_relu_derivative(self.z[i-1])
                elif self.activations[i-1] == ActivationFunction.relu:
                    dz = np.dot(dz, self.weights[i].T) * ActivationFunction.relu_derivative(self.z[i-1])
                elif self.activations[i-1] == ActivationFunction.softmax:
                    dz = np.dot(dz, self.weights[i].T) * ActivationFunction.softmax_derivative(self.z[i-1])
    
    def initialize_adam(self):
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def update_weights(self, optimizer='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8):
        if optimizer == 'adam':
            if not hasattr(self, 'm'):
                self.initialize_adam()

            self.t += 1
            lr_t = self.learning_rate

            for i in range(len(self.weights)):
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * self.weights_gradients[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (self.weights_gradients[i] ** 2)
                m_hat = self.m[i] / (1 - beta1 ** self.t)
                v_hat = self.v[i] / (1 - beta2 ** self.t)
                self.weights[i] -= lr_t * m_hat / (np.sqrt(v_hat) + epsilon)

                self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * self.biases_gradients[i]
                self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (self.biases_gradients[i] ** 2)
                m_hat_b = self.m_b[i] / (1 - beta1 ** self.t)
                v_hat_b = self.v_b[i] / (1 - beta2 ** self.t)
                self.biases[i] -= lr_t * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
        else:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * self.weights_gradients[i]
                self.biases[i] -= self.learning_rate * self.biases_gradients[i]
    
    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)
    
    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            self.weights, self.biases = pickle.load(f)
    
    def train(self, X, y, X_val, y_val, epochs, learning_rate=0.1, optimizer='sgd', generate_new_params=True):
        self.learning_rate = learning_rate
        if generate_new_params:
            self.weights, self.biases = [], []
            self.initialize_parameters()
            print('Params rewritten')

        error_log = []
        val_error_log = []
        patience_counter = 0
        patience = 10
        flag = True

        if optimizer == 'adam':
            self.initialize_adam()

        for epoch in range(epochs):
            y_pred = self.forward_propagation(X)
            loss = self.cross_entropy_loss(y, y_pred)

            self.backward_propagation(X, y, y_pred)
            self.update_weights(optimizer)
            error_log.append(loss)

            # Early stopping break conditions
            y_val_pred = self.forward_propagation(X_val)
            val_loss = self.cross_entropy_loss(y_val, y_val_pred)
            val_error_log.append(val_loss)

            if epoch == 0:
                prev_val_loss = val_loss
            else:
                if val_loss > prev_val_loss:
                    if flag:
                        self.save_weights('best_weights.pkl')
                        flag = False
                    patience_counter += 1
                else:
                    patience_counter = 0
                    flag = True
                prev_val_loss = val_loss

            if patience_counter > patience:
                print(f'Early stopping at Epoch: {epoch}, Patience: {patience_counter}')
                break

            print(f'Epoch {epoch}, Training Loss: {loss:.3e}, Validation Loss: {val_loss:.3e}, Patience: {patience_counter}, Learning Rate: {self.learning_rate}')

        if epoch == epochs - 1 or patience_counter > patience:
            print(f'Epoch {epoch}, Training Loss: {loss:.3e}, Validation Loss: {val_loss:.3e}, Patience: {patience_counter}')
            if patience_counter > patience:
                self.load_weights('best_weights.pkl')
            return error_log, val_error_log
        else: 
            self.weights, self.biases = [], []
            self.initialize_parameters()
            print('Reinitializing Param...')

    def predict(self, X):
        return self.forward_propagation(X)
    
    def test(self, X_test, y_test):
        y_pred = self.predict(X_test)
        loss = self.cross_entropy_loss(y_test, y_pred)
        print(f'Test Loss: {loss}')
        return y_pred

    def plot_training_error(self, train_losses, val_losses=None):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        if val_losses is not None:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
    
    def add_labels_from_folders(self, folder_path):
        self.labels = []
        for label in sorted(os.listdir(folder_path)):
            if os.path.isdir(os.path.join(folder_path, label)):
                self.labels.append(label)
        self.output_size = len(self.labels)

    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = np.argmax(self.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = np.zeros((self.output_size, self.output_size), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(self.output_size)
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.show()

    def evaluate_metrics(self, X_test, y_test):
        y_pred = np.argmax(self.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        precision = np.zeros(self.output_size)
        recall = np.zeros(self.output_size)
        f1 = np.zeros(self.output_size)
        for i in range(self.output_size):
            tp = np.sum((y_pred == i) & (y_true == i))
            fp = np.sum((y_pred == i) & (y_true != i))
            fn = np.sum((y_pred != i) & (y_true == i))
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        print(f'Accuracy: {accuracy}')
        for i, label in enumerate(self.labels):
            print(f'{label} - Precision: {precision[i]}, Recall: {recall[i]}, F1 Score: {f1[i]}')
        print(f'Mean Precision: {np.mean(precision)}')
        print(f'Mean Recall: {np.mean(recall)}')
        print(f'Mean F1 Score: {np.mean(f1)}')

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

