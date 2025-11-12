# -*- coding: utf-8 -*-
"""
EEG-based Cognitive Task Classification using CNN
-------------------------------------------------
This script loads EEG data from MATLAB .mat files, preprocesses them,
and trains a convolutional neural network (CNN) to classify TMT-A vs. TMT-B tasks.
"""

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ======================================================
# 1. Load EEG Data (.mat)
# ======================================================

# Example: iPad-TMT (0.5–60 Hz)
data1 = scipy.io.loadmat(r'C:\Matlab Working Path\edata\ipad-TMT-re-filter(.mat_file)_data\subj1_task1.mat')
data2 = scipy.io.loadmat(r'C:\Matlab Working Path\edata\ipad-TMT-re-filter(.mat_file)_data\subj1_task2.mat')

# Alternative data sources (uncomment as needed)
# ------------------------------------------------------
# PC-TMT data:
# data1 = scipy.io.loadmat(r'C:\Matlab Working Path\edata\PC-easy-hard(.mat_file)_data\subj1_task1.mat')
# data2 = scipy.io.loadmat(r'C:\Matlab Working Path\edata\PC-easy-hard(.mat_file)_data\subj1_task2.mat')

# Combined 8-subject data (0.1–30 Hz):
# data1 = scipy.io.loadmat(r'C:\Matlab Working Path\edata\ipad-TMT-re-filter(.mat_file)_data_combined\allTask1Chunks.mat')
# data2 = scipy.io.loadmat(r'C:\Matlab Working Path\edata\ipad-TMT-re-filter(.mat_file)_data_combined\allTask2Chunks.mat')

# 28-electrode extracted data:
# data1 = scipy.io.loadmat(r'C:\Matlab Working Path\edata\Extract_important_electrode\extract_ipad-TMT-re-filter(.mat_file)_data\subj8_task1.mat')
# data2 = scipy.io.loadmat(r'C:\Matlab Working Path\edata\Extract_important_electrode\extract_ipad-TMT-re-filter(.mat_file)_data\subj8_task2.mat')


# ======================================================
# 2. Data Preparation
# ======================================================

chunks1 = data1['chunks']
chunks2 = data2['chunks']

# Convert MATLAB structures to NumPy arrays
data_chunks1 = np.array([chunks1[i][0] for i in range(chunks1.shape[0])])
data_chunks2 = np.array([chunks2[i][0] for i in range(chunks2.shape[0])])

print('Task 1 shape:', data_chunks1.shape)
print('Task 2 shape:', data_chunks2.shape)

# Create binary labels
labels1 = np.zeros(len(data_chunks1))  # Task 1 → 0
labels2 = np.ones(len(data_chunks2))   # Task 2 → 1

# Merge data and labels
X = np.concatenate((data_chunks1, data_chunks2), axis=0)
y = np.concatenate((labels1, labels2), axis=0)

print('Combined X shape:', X.shape)
print('Combined y shape:', y.shape)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors (add channel dimension)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)


# ======================================================
# 3. Define CNN Model
# ======================================================

class EEG_CNN(nn.Module):
    def __init__(self):
        super(EEG_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout_conv = nn.Dropout2d(0.5)
        self.dropout_fc = nn.Dropout(0.5)
        
        # Adjust input size depending on electrode count
        self.fc1 = nn.Linear(64 * 7 * 62, 64)   # For 56 electrodes
        # self.fc1 = nn.Linear(64 * 3 * 62, 64)  # For 28 electrodes
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.dropout_conv(F.relu(self.conv2(x))))
        x = self.pool(self.dropout_conv(F.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x


# ======================================================
# 4. Training Setup
# ======================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEG_CNN().to(device)
print(f'Using device: {device}')

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100
best_val_accuracy = 0.0
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []


# ======================================================
# 5. Training Loop
# ======================================================

for epoch in range(num_epochs):
    # ----- Training -----
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ----- Validation -----
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    preds, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Save best model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_state = model.state_dict()
        best_preds, best_labels = preds, labels_all

    print(f'Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

# ======================================================
# 6. Evaluation and Visualization
# ======================================================

model.load_state_dict(best_state)
best_preds = np.array(best_preds)
best_labels = np.array(best_labels)

print(f'\nBest Validation Accuracy: {best_val_accuracy:.2f}%')
print('Confusion Matrix:\n', confusion_matrix(best_labels, best_preds))
print('Classification Report:\n', classification_report(best_labels, best_preds, digits=4))

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss over Epochs'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy over Epochs'); plt.legend()

plt.tight_layout()
plt.show()
