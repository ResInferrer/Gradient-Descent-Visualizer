from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import gdv_vis
from .data_generator import DataGenerator


class Runner():
    def __init__(self):
        self.n_samples = 100
        self.n_features = 20
        self.n_classes = 2

    def run(self):
        """
        Generates a dataset with features and target values, processes, predicts, 
        and also launches visualization of these data (points) using matplotlib. 

        The calculation logic is located in the cpp module.
        """

        print("\n===========================")
        print("Gradient Descent Visualizer")
        print("===========================\n")

        X, y = DataGenerator.dataset_generation(
            self.n_samples,
            self.n_features,
            self.n_classes
        )

        X_part, X_val, y_part, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Select model: logistic regression (0) or Support Vector Machine (SVM) (1)")

        choice = int(input("Choice (int): "))
        while True:
            if choice == 0:
                print("\nLogistic Regrssion\n")
                self.logistic_regression()
                break
            elif choice == 1:
                print("\nSupport Vector Machine (SVM)\n")
                self.svm()
                break
            else:
                print("No, we need numbers here.")
                print("Write 0 if you need to use logistic regression,")
                print("or 1 if you need Support Vector Machine (SVM)\n")
                choice = int(input("Choice (int): "))



    def logistic_regression(self):
        """
        
        """
        pass

    def svm(self):
        """
        
        """
        pass


