from sklearn.model_selection import train_test_split

from .data_generator import DataGenerator
from .models import Models

class Runner():
    def __init__(self):
        # Experiment parameters – you can modify these for testing
        self.n_samples = 1000      # Number of data points
        self.n_features = 200      # Number of features (dimensions)
        self.n_classes = 2         # Number of classes (2 for binary, >2 for multiclass)


    def run(self):
        """
        Generates a dataset with features and target values, processes, predicts, 
        and also launches visualization of these data (points) using matplotlib. 

        The calculation logic is located in the cpp module.
        """

        print("\n" + "="*40)
        print("Gradient Descent Visualizer")
        print("="*40 + "\n")

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
                print("\n" + "-"*40)
                print("Logistic Regrssion")
                print("-"*40 + "\n")

                print("Select problem type: binary (0) or multiclass One-vs-All (1)")
                problem_type = int(input("Choice (int): "))
                if problem_type == 0:
                    Models.logistic_regression_binary(self, X_part, X_val, y_part, y_val)
                else:
                    Models.logistic_regression_multiclass(self, X_part, X_val, y_part, y_val)

                break
            elif choice == 1:
                print("\n" + "-"*40)
                print("\nSupport Vector Machine (SVM)\n")
                print("-"*40 + "\n")
                Models.svm(self, X_part, X_val, y_part, y_val)
                break
            else:
                print("No, we need numbers here.")
                print("Write 0 if you need to use logistic regression,")
                print("or 1 if you need Support Vector Machine (SVM)\n")
                choice = int(input("Choice (int): "))

