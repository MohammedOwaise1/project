import joblib
import numpy as np
from features import extract_features

classes = ['plastic', 'organic', 'metal', 'paper', 'glass']

def predict_waste(image_path):
    clf = joblib.load('model.pkl')

    features = extract_features(image_path)
    features = np.array(features).reshape(1, -1)

    prediction = clf.predict(features)
    predicted_class = classes[prediction[0]]

    print(f"Predicted Waste Type: {predicted_class}")

if __name__ == '__main__':
    image_path = input("Enter path to waste image: ")
    predict_waste(image_path)
