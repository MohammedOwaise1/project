from train import train_model
from predict import predict_waste
import os

def main():
    while True:
        print("\nWaste Segregation System (Using Python 3.13 Compatible Libraries)")
        print("1. Train Model")
        print("2. Predict Waste Type")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            train_model()

        elif choice == "2":
            image_path = input("Enter path to waste image: ")
            if os.path.exists(image_path):
                predict_waste(image_path)
            else:
                print("Image not found!")

        elif choice == "3":
            break

        else:
            print("Invalid choice, try again.")

if __name__ == '__main__':
    main()
