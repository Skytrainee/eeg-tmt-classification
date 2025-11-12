# -*- coding: utf-8 -*-
"""
EEG CNN-LSTM Classification for TMT Tasks
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

# =====================================================
#                Load EEG Data (.mat)
# =====================================================

# Example: Paper-pencil TMT, delta band, combined data
data1 = scipy.io.loadmat('C:\\Matlab Working Path\\edata\\pp_TMT_data\\pp-delta-combined\\allTask1Chunks.mat')
data2 = scipy.io.loadmat('C:\\Matlab Working Path\\edata\\pp_TMT_data\\pp-delta-combined\\allTask2Chunks.mat')

# Extract data arrays from .mat structure
chunks1 = data1['allTask1Chunks']
chunks2 = data2['allTask2Chunks']

# Convert to NumPy arrays
data_chunks1 = np.array([chunks1[i][0] for i in range(chunks1.shape[0])])
data_chunks2 = np.array([chunks2[i][0] for i in range(chunks2.shape[0])])

print("Data shapes:")
print("Task1:", data_chunks1.shape)
print("Task2:", data_chunks2.shape)

# Create labels: 0 for Task1, 1 for Task2
labels1 = np.zeros(len(data_chunks1))
labels2 = np.ones(len(data_chunks2))

# Concatenate data and labels
X = np.concatenate((data_chunks1, data_chunks2), axis=0)
y = np.concatenate((labels1, labels2), axis=0)

print("Merged data shape:", X.shape)
print("Labels shape:", y.shape)

# =====================================================
#                Train/Validation Split
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # add channel dimension
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =====================================================
#                Define CNN-LSTM Model
# =====================================================

class EEG_CNN_LSTM(nn.Module):
    def __init__(self):
        super(EEG_CNN_LSTM, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.dropout_conv = nn.Dropout2d(0.5)

        # Adjust this size according to EEG electrode and sample configuration
        H_prime, W_prime = 7, 62  # For 56 electrodes
        # H_prime, W_prime = 3, 62  # For 28 electrodes
        self.fc_input_size = 64 * H_prime * W_prime

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.fc_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        # Fully connected layer
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN feature extraction
        c_out = self.pool(F.relu(self.conv1(x)))
        c_out = self.pool(self.dropout_conv(F.relu(self.conv2(c_out))))
        c_out = self.pool(self.dropout_conv(F.relu(self.conv3(c_out))))

        # Reshape for LSTM input
        r_in = c_out.view(batch_size, 1, -1)

        # LSTM forward
        self.lstm.flatten_parameters()
        r_out, _ = self.lstm(r_in)

        # Take the last time step
        r_out_last = r_out[:, -1, :]

        # Fully connected + sigmoid output
        out = torch.sigmoid(self.fc1(r_out_last))
        return out


# =====================================================
#                Model Initialization
# =====================================================
model = EEG_CNN_LSTM()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# =====================================================
#                Training and Validation
# =====================================================
num_epochs = 100
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_accuracy = 0.0
best_model_state = None
best_preds, best_labels = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs).squeeze()
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)

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

    # Validation
    model.eval()
    val_loss_total, val_correct, val_total = 0.0, 0, 0
    current_preds, current_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item()
            predicted = (outputs > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            current_preds.extend(predicted.cpu().numpy())
            current_labels.extend(labels.cpu().numpy())

    val_loss = val_loss_total / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Save best model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = model.state_dict()
        best_preds = current_preds
        best_labels = current_labels

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# =====================================================
#                Evaluation and Visualization
# =====================================================
if best_model_state is not None:
    model.load_state_dict(best_model_state)

best_preds = np.array(best_preds)
best_labels = np.array(best_labels)

# Classification metrics
accuracy = accuracy_score(best_labels, best_preds)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(best_labels, best_preds)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(best_labels, best_preds, digits=5)
print("Classification Report:\n", class_report)

print(f"Highest Validation Accuracy: {best_val_accuracy:.2f}%")

# =====================================================
#                Plot Training Curves
# =====================================================
plt.figure(figsize=(12, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
