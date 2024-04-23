import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import os
from fashion_mnist.utils import mnist_reader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def to_one_hot(labels, num_classes):
    num_samples = len(labels)
    one_hot_matrix = np.zeros((num_samples, num_classes))
    for i, label in enumerate(labels):
        one_hot_matrix[i, label] = 1
    return one_hot_matrix

def cross_entropy(predictions, labels):

    batch_size = predictions.shape[0]
    correct_log_probs = -np.log(predictions[range(batch_size), labels]+1e-12)
    loss = np.sum(correct_log_probs) / batch_size
    return loss

def d_cross_entropy(predictions, labels):
    if labels.shape != predictions.shape:
        # labels not one-hot
        labels = to_one_hot(labels, predictions.shape[1])
    return predictions - labels

class Classifier:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        if self.activation == 'relu':
            self.A1 = relu(self.Z1)
        else:
            raise ValueError("Unsupported activation function")
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return softmax(self.Z2)

    def compute_loss(self, Y_hat, Y, reg_lambda=0):
        return cross_entropy(Y_hat, Y) + 0.5 * reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))

    def backward(self, X, Y, Y_hat, reg_lambda=0):
        batch_size = Y.shape[0]
        dZ2 = d_cross_entropy(Y_hat, Y)
        dW2 = np.dot(self.A1.T, dZ2) / batch_size + reg_lambda * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        if self.activation == 'relu':
            dZ1 = np.dot(dZ2, self.W2.T) * d_relu(self.Z1)
        else:
            raise ValueError("Unsupported activation function")
        dW1 = np.dot(X.T, dZ1) / batch_size + reg_lambda * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return
    
    def load_model(self,path):
        paras = np.load(path)
        self.W1 = paras['W1']
        self.b1 = paras['b1']
        self.W2 = paras['W2']
        self.b2 = paras['b2']
        
    def save_model(self,path):
        if not os.path.exists(path):
            os.mknod(path)
        np.savez(path, W1 = self.W1, b1 = self.b1, W2 = self.W2, b2 = self.b2)
        
    def print_parameters(self):
        print('W1',self.W1.shape)
        print(self.W1)
        print('b1',self.b1.shape)
        print(self.b1)
        print('W2',self.W2.shape)
        print(self.W2)
        print('b2',self.b2.shape)
        print(self.b2)
        
def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr, reg_lambda, lr_decay_rate):
    best_val_acc = 0
    best_weights = None
    n_batches = int(np.ceil(X_train.shape[0] / batch_size))
    val_accs = []
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        total_train_loss = 0
        X_train, y_train = shuffle(X_train, y_train)
        for batch in range(n_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, X_train.shape[0])
            inputs= X_train[start:end]
            labels = y_train[start:end]
            predictions = model.forward(inputs)
            
            loss = model.compute_loss(predictions, labels, reg_lambda)
            total_train_loss += loss
            gradients = model.backward(inputs, labels, predictions, reg_lambda)
          
            model.update_params(*gradients, lr)

        # Decay learning rate
        lr *= lr_decay_rate

        # Validation accuracy
        val_predictions = model.forward(X_val)
        val_loss = model.compute_loss(val_predictions, y_val, reg_lambda)
        val_acc = accuracy_score(y_val, np.argmax(val_predictions, axis=1))
        train_losses.append(total_train_loss / n_batches)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Epoch {epoch + 1}, Train Loss: {loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = "./models/Classifier_best.npz"
            model.save_model(path)
            # Save model
        if epoch % 5 == 4:
            path = f"./models/Classifier_epoch{epoch}.npz"
            model.save_model(path)
    
    return train_losses,val_losses,val_accs

def test(model,X_test,y_test):
    predictions = model.forward(X_test)
    return accuracy_score(y_test, np.argmax(predictions, axis=1))

def save_results(train_losses,val_losses,val_accs,path):
    if not os.path.exists(path):
        os.mkdir(path)
    epochs = range(1, len(train_losses) + 1)
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Training Loss')
    ax1.plot(epochs, val_losses, color='tab:red', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, val_accs, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.savefig(os.path.join(path,"./result.png"))
    plt.show()
    import json
    dic = {
        "train_losses":train_losses,
        "val_losses":val_losses,
        "val_accs":val_accs
    }
    with open(os.path.join(path,"./result.json"),"w") as f:
        json.dump(dic,f,indent=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_training', default=True, type=bool, choices=[True,False], help='Train or not')
    parser.add_argument('--do_testing', default=True, type=bool, choices=[True,False], help='Test or not')
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--reg_lambda', default=1e-3, type=float)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=0.95, type=float)
    
    
    args = parser.parse_args()
    do_training = args.do_training
    do_testing = args.do_testing

    # Model setting
    input_size = 784 
    hidden_size = args.hidden_size
    output_size = 10
    model = Classifier(input_size, hidden_size, output_size)

    if do_training == True:
        # Hyper parameters:
        epochs = args.epochs
        batch_size = args.batch_size
        reg_lambda = args.reg_lambda
        lr = args.learning_rate
        lr_decay_rate = args.lr_decay_rate
        # Training
        X, y = mnist_reader.load_mnist('data/fashion', kind='train')
        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1,random_state=51)
        train_losses,val_losses,val_accs = train(model, X_train, y_train, X_val, y_val, 
            epochs=epochs, batch_size=batch_size, lr=lr, reg_lambda=reg_lambda, lr_decay_rate=lr_decay_rate,
            )
        log_path = f"./log/Acc={val_accs[-1]}_hiddenSize={hidden_size}_batchSize={batch_size}_lr={lr}_lambda={reg_lambda}_decayRate={lr_decay_rate}/"
        
        
        
    if do_testing == True:
        # Testing
        X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
        path = "./models/Classifier_best.npz"
        model.load_model(path)
        test_acc = test(model,X_test,y_test)
        print("Test Accuracy: ",test_acc)
        log_path = f"./log/Acc={test_acc}_hiddenSize={hidden_size}_batchSize={batch_size}_lr={lr}_lambda={reg_lambda}_decayRate={lr_decay_rate}/"
    
    if do_training == True:
        save_results(train_losses,val_losses,val_accs,log_path)