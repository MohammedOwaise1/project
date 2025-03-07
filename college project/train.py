import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from features import extract_features

# Define classes
classes = ['plastic', 'organic', 'metal', 'paper', 'glass']

def train_model():
    X = []
    y = []

    base_dir = 'data'

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Directory '{class_dir}' not found. Skipping.")
            continue

        for file_name in os.listdir(class_dir):
            if file_name.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(class_dir, file_name)

                # Extract features
                features = extract_features(image_path)
                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save model
    joblib.dump(clf, 'model.pkl')
    print("Model trained and saved as 'model.pkl'")

if __name__ == '__main__':

    train_model()


