import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set paths
DATA_DIR = r'C:\Users\BOSS\Desktop\PRODIGY_ML_03\train\dogs and cats'  # Folder containing the images
IMG_SIZE = 64        # Resize images to 64x64

def load_data(data_dir, img_size=64, limit=2500):
    X, y = [], []
    count = {'cat': 0, 'dog': 0}
    for img_name in tqdm(os.listdir(data_dir)):
        label = 0 if 'cat' in img_name else 1
        if count['cat'] >= limit and label == 0:
            continue
        if count['dog'] >= limit and label == 1:
            continue
        path = os.path.join(data_dir, img_name)
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img.flatten())
            y.append(label)
            if label == 0:
                count['cat'] += 1
            else:
                count['dog'] += 1
        except:
            continue
        if count['cat'] >= limit and count['dog'] >= limit:
            break
    return np.array(X), np.array(y)

# Load and prepare data
print("[INFO] Loading data...")
X, y = load_data(DATA_DIR, IMG_SIZE, limit=1500)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("[INFO] Training SVM...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate
print("[INFO] Evaluating...")
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Plot some predictions
def plot_predictions(X, y_true, y_pred, img_size, save_path="svm_results.png"):
    plt.figure(figsize=(10, 6))
    for i in range(6):
        img = X[i].reshape(img_size, img_size, 3).astype('uint8')
        true_label = 'Cat' if y_true[i] == 0 else 'Dog'
        pred_label = 'Cat' if y_pred[i] == 0 else 'Dog'
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Example usage
plot_predictions(X_test, y_test, y_pred, IMG_SIZE)
