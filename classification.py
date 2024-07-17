#1
 import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a simple logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

# Hyperparameters
input_size = 28 * 28  # MNIST images are 28x28 pixels
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, 
download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, 
download=True, transform=transform)

# Split the train_dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create DataLoaders for each set
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function and optimizer
model = LogisticRegressionModel(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], 
            Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
#2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model.eval()  # Set the model to evaluation mode

# Helper function to calculate metrics
def calculate_metrics(loader):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return accuracy, precision, recall, f1

train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_loader)
val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(val_loader)
test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_loader)

print(f'Training set - Accuracy: {train_accuracy}, Precision: {train_precision},
Recall: {train_recall}, F1 Score: {train_f1}')
print(f'Validation set - Accuracy: {val_accuracy}, Precision: {val_precision}, 
Recall: {val_recall}, F1 Score: {val_f1}')
print(f'Test set - Accuracy: {test_accuracy}, Precision: {test_precision}, 
Recall: {test_recall}, F1 Score: {test_f1}')

#3
from sklearn.model_selection import ParameterGrid

# Define parameter grid
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'batch_size': [32, 64, 128],
}

best_val_accuracy = 0
best_params = {}

for params in ParameterGrid(param_grid):
    # Create DataLoader with current batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'],
    shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=params['batch_size'],
    shuffle=False)
    
    # Initialize the model, loss function and optimizer with current learning rate
    model = LogisticRegressionModel(input_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Train the model
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate the model on validation set
    val_accuracy, _, _, _ = calculate_metrics(val_loader)
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = params

print(f'Best parameters: {best_params}')
print(f'Best validation accuracy: {best_val_accuracy}')

#4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get data for visualization
def get_data_for_visualization(loader):
    images, labels = next(iter(loader))
    images = images.reshape(-1, 28*28).numpy()
    labels = labels.numpy()
    return images, labels

# Reduce dimensionality with PCA
pca = PCA(n_components=2)
train_images, train_labels = get_data_for_visualization(train_loader)
pca_result = pca.fit_transform(train_images)

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1)) 
    Z = model(torch.tensor(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]),
    dtype=torch.float32))
    Z = torch.max(Z.data, 1)[1].numpy()
    Z = Z.reshape(xx.shape)  
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Decision Boundary')
    plt.show()

# Train the model again for plotting (with best parameters from grid search)
model = LogisticRegressionModel(input_size, num_classes)
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
train_loader = DataLoader(dataset=train_dataset, batch_size=best_params['batch_size'], 
shuffle=True)
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Plot the decision boundary
plot_decision_boundary(model, pca_result, train_labels)
