import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# I. iPad-TMT (uncomment the section you need)
# ============================================

# Example paths for different filtering and datasets
# subj1task1_file = "C:\\Matlab Working Path\\edata\\ipad-TMT-a-b(.mat_file)_data\\subj8_task1.mat"
# subj1task2_file = "C:\\Matlab Working Path\\edata\\ipad-TMT-a-b(.mat_file)_data\\subj8_task2.mat"

# 0.5–60 Hz filtered data
# subj1task1_file = "C:\\Matlab Working Path\\edata\\ipad-TMT-re-filter(.mat_file)_data\\subj8_task1.mat"
# subj1task2_file = "C:\\Matlab Working Path\\edata\\ipad-TMT-re-filter(.mat_file)_data\\subj8_task2.mat"

# Delta-band filtered data
# subj1task1_file = "C:\\Matlab Working Path\\edata\\filter_delta_ipad-TMT-re-filter(.mat_file)_data\\delta_subj8_task1.mat"
# subj1task2_file = "C:\\Matlab Working Path\\edata\\filter_delta_ipad-TMT-re-filter(.mat_file)_data\\delta_subj8_task2.mat"

# Combined data for all subjects
# subj1task1_file = "C:\\Matlab Working Path\\edata\\ipad-TMT-re-filter(.mat_file)_data_combined\\allTask1Chunks.mat"
# subj1task2_file = "C:\\Matlab Working Path\\edata\\ipad-TMT-re-filter(.mat_file)_data_combined\\allTask2Chunks.mat"

# 28-electrode extracted data
# subj1task1_file = "C:\\Matlab Working Path\\edata\\Extract_important_electrode\\extract_ipad-TMT-re-filter(.mat_file)_data\\subj1_task1.mat"
# subj1task2_file = "C:\\Matlab Working Path\\edata\\Extract_important_electrode\\extract_ipad-TMT-re-filter(.mat_file)_data\\subj1_task2.mat"


# ============================================
# II. PP-TMT or VR-TMT
# ============================================

# PP-TMT (0.5–60 Hz)
subj1task1_file = "C:\\Matlab Working Path\\edata\\pp_TMT_data\\0.5-60HZ\\subj8_task1.mat"
subj1task2_file = "C:\\Matlab Working Path\\edata\\pp_TMT_data\\0.5-60HZ\\subj8_task2.mat"

# VR-TMT (examples)
# subj1task1_file = "C:\\Matlab Working Path\\edata\\VR_TMT_data\\0.5-60HZ\\subj1_task1.mat"
# subj1task2_file = "C:\\Matlab Working Path\\edata\\VR_TMT_data\\0.5-60HZ\\subj1_task2.mat"
# subj1task1_file = "C:\\Matlab Working Path\\edata\\VR_TMT_data\\VR-delta\\delta_subj8_task1.mat"
# subj1task2_file = "C:\\Matlab Working Path\\edata\\VR_TMT_data\\VR-delta\\delta_subj8_task2.mat"


# ============================================
# Function: Load MATLAB data
# ============================================
def load_data(file, var_name):
    """Load a specific variable from a .mat file."""
    mat_data = scipy.io.loadmat(file)
    return mat_data[var_name]


# Load data for Task 1 and Task 2
task1_data = load_data(subj1task1_file, 'chunks')
task2_data = load_data(subj1task2_file, 'chunks')

# Alternate variable names if needed:
# task1_data = load_data(subj1task1_file, 'filtered_chunks')
# task2_data = load_data(subj1task2_file, 'filtered_chunks')
# task1_data = load_data(subj1task1_file, 'extracted_chunks')
# task2_data = load_data(subj1task2_file, 'extracted_chunks')

# ============================================
# Verify and preprocess data
# ============================================
print(f"Task 1 data shape: {task1_data.shape}")
print(f"Task 2 data shape: {task2_data.shape}")
print(f"Example cell shape (Task 1): {task1_data[0][0].shape}")
print(f"Example cell shape (Task 2): {task2_data[0][0].shape}")

# Flatten each EEG chunk into one vector per sample
X_task1 = np.array([cell[0].flatten() for cell in task1_data])
X_task2 = np.array([cell[0].flatten() for cell in task2_data])

# Create binary labels
y_task1 = np.zeros(X_task1.shape[0])
y_task2 = np.ones(X_task2.shape[0])

# Combine data and labels
X = np.vstack((X_task1, X_task2))
y = np.concatenate((y_task1, y_task2))

print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

# ============================================
# Feature standardization
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled features shape: {X_scaled.shape}")

# ============================================
# Classification using SVM
# ============================================
svm = SVC(kernel='rbf', random_state=42)

# 5-fold cross-validation
cross_val_scores = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cross_val_scores}")
print(f"Mean cross-validation accuracy: {np.mean(cross_val_scores):.5f}")

# ============================================
# Train-test split evaluation
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# ============================================
# Evaluation metrics
# ============================================
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=5))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
