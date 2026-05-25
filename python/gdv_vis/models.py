import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import gdv_vis
from .visualizer import Visualizer

class Models():
    def logistic_regression_binary(self, X_part, X_val, y_part, y_val):
        """Train and evaluate logistic regression from C++ module"""

        unique_classes = len(set(y_part))
        if unique_classes != 2:
            print(f"Error: Binary logistic regression requires exactly 2 classes, but found {unique_classes} classes.")
            print("   Please use multiclass mode.\n")
            return

        # Parameters for training
        learning_rate = 0.01
        iterations = 1000
        l2_lambda = 0.01
        threshold = 0.5

        print(f"Parameters:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Iterations: {iterations}")
        print(f"  - L2 regularization: {l2_lambda}")
        print(f"  - Threshold: {threshold}\n")

        model = gdv_vis.LogisticRegression()
        
        # Model training
        model.fit(X_part, y_part, learning_rate, iterations, l2_lambda, verbose=True)
        
        # Predictions on train and validation
        y_train_pred = []
        for row in X_part:
            pred = model.predict(row, threshold)
            y_train_pred.append(pred)
        
        y_val_pred = []
        for row in X_val:
            pred = model.predict(row, threshold)
            y_val_pred.append(pred)
        
        # Metrics for train
        print("TRAIN SET METRICS (binary):")
        print(f"Accuracy:  {accuracy_score(y_part, y_train_pred):.4f}")
        print(f"Precision: {precision_score(y_part, y_train_pred, average='binary'):.4f}")
        print(f"Recall:    {recall_score(y_part, y_train_pred, average='binary'):.4f}")
        print(f"F1-score:  {f1_score(y_part, y_train_pred, average='binary'):.4f}")
        print("\n")

        # Metrics for validation
        print("VALIDATION SET METRICS (binary):")
        print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
        print(f"Precision: {precision_score(y_val, y_val_pred, average='binary'):.4f}")
        print(f"Recall:    {recall_score(y_val, y_val_pred, average='binary'):.4f}")
        print(f"F1-score:  {f1_score(y_val, y_val_pred, average='binary'):.4f}")
        print("\n")


        # Visualization
        print("\nGenerating visualizations...")
        
        # Create metrics dictionaries
        metrics_train = {
            'Accuracy': accuracy_score(y_part, y_train_pred),
            'Precision': precision_score(y_part, y_train_pred, average='binary'),
            'Recall': recall_score(y_part, y_train_pred, average='binary'),
            'F1-score': f1_score(y_part, y_train_pred, average='binary')
        }
        metrics_val = {
            'Accuracy': accuracy_score(y_val, y_val_pred),
            'Precision': precision_score(y_val, y_val_pred, average='binary'),
            'Recall': recall_score(y_val, y_val_pred, average='binary'),
            'F1-score': f1_score(y_val, y_val_pred, average='binary')
        }
        
        # Generate full report
        classes = [0, 1]
        Visualizer.create_full_report(
            X_part, y_part, y_train_pred,
            X_val, y_val, y_val_pred,
            metrics_train, metrics_val,
            classes=classes,
            save_path="graphs/logistic_regression_binary.png"
        )
        
        # Also show individual plots (optional)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        Visualizer.plot_metrics_bar(metrics_train, title="Train Metrics")
        plt.subplot(1, 2, 2)
        Visualizer.plot_metrics_bar(metrics_val, title="Validation Metrics")
        plt.tight_layout()
        plt.show()
        
        print("Visualizations saved and displayed.")


    def logistic_regression_multiclass(self, X_train, X_val, y_train, y_val):
        """
        Multiclass logistic regression using One-vs-All strategy.
        Assumes y_train and y_val contain integer labels 0, 1, ..., n_classes-1.
        """

        # Defining Unique Classes
        classes = sorted(set(y_train))
        n_classes = len(classes)
        print(f"\nDetected classes: {classes} (total {n_classes} classes)")
        
        # Parameters for training
        learning_rate = 0.01
        iterations = 1000
        l2_lambda = 0.01

        print(f"Parameters:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Iterations: {iterations}")
        print(f"  - L2 regularization: {l2_lambda}")

        # Model training
        models = []
        for cls in classes:
            print(f"\nTraining classifier for class {cls} vs rest...")
            # Transform the labels: 1 for the current class, 0 for all others
            y_binary = [1 if label == cls else 0 for label in y_train]
            
            model = gdv_vis.LogisticRegression()
            model.fit(X_train, y_binary, learning_rate, iterations, l2_lambda, verbose=False)
            models.append((cls, model))

        print("\n")

        def predict_proba_all(X, models):
            """Returns a probability matrix [n_samples, n_classes]"""
            n_samples = len(X)
            n_classes = len(models)
            proba = [[0.0]*n_classes for _ in range(n_samples)]
            for i, (cls, model) in enumerate(models):
                for j, row in enumerate(X):
                    p = model.predict_proba(row)[1]
                    proba[j][i] = p
            return proba

        def predict_classes(X, models, classes):
            proba = predict_proba_all(X, models)
            preds = []
            for row_proba in proba:
                max_idx = max(range(len(row_proba)), key=lambda i: row_proba[i])
                preds.append(classes[max_idx])
            return preds
        
        y_train_pred = predict_classes(X_train, models, classes)
        y_val_pred = predict_classes(X_val, models, classes)

        print("TRAIN SET METRICS (multiclass):")
        print(f"Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
        print(f"Precision: {precision_score(y_train, y_train_pred, average='macro'):.4f}")
        print(f"Recall:    {recall_score(y_train, y_train_pred, average='macro'):.4f}")
        print(f"F1-score:  {f1_score(y_train, y_train_pred, average='macro'):.4f}")
        print("\n")

        print("VALIDATION SET METRICS (multiclass):")
        print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
        print(f"Precision: {precision_score(y_val, y_val_pred, average='macro'):.4f}")
        print(f"Recall:    {recall_score(y_val, y_val_pred, average='macro'):.4f}")
        print(f"F1-score:  {f1_score(y_val, y_val_pred, average='macro'):.4f}")
        print("\n")


        # Visualization
        print("\nGenerating visualizations...")
        metrics_train = {
            'Accuracy': accuracy_score(y_train, y_train_pred),
            'Precision': precision_score(y_train, y_train_pred, average='macro'),
            'Recall': recall_score(y_train, y_train_pred, average='macro'),
            'F1-score': f1_score(y_train, y_train_pred, average='macro')
        }
        metrics_val = {
            'Accuracy': accuracy_score(y_val, y_val_pred),
            'Precision': precision_score(y_val, y_val_pred, average='macro'),
            'Recall': recall_score(y_val, y_val_pred, average='macro'),
            'F1-score': f1_score(y_val, y_val_pred, average='macro')
        }
        # Full multi-plot report (works for multiclass)
        Visualizer.create_full_report(
            X_train, y_train, y_train_pred,
            X_val, y_val, y_val_pred,
            metrics_train, metrics_val,
            classes=classes,
            save_path="graphs/logistic_regression_multiclass.png"
        )

        # Additional bar charts
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        Visualizer.plot_metrics_bar(metrics_train, title="Train Metrics")
        plt.subplot(1, 2, 2)
        Visualizer.plot_metrics_bar(metrics_val, title="Validation Metrics")
        plt.tight_layout()
        plt.show()

        print("Visualizations saved and displayed.")


    def svm(self, X_train, X_val, y_train, y_val):
        """Train and evaluate SVM from C++ module (binary classification)."""
        # Check number of classes (SVM is binary)
        unique_classes = len(set(y_train))
        if unique_classes != 2:
            print(f"Error: SVM requires exactly 2 classes, but found {unique_classes} classes.")
            print("   Only binary classification is supported for now.\n")
            return

        # Convert labels from {0,1} to {-1,1} as required by C++ SVM
        y_train_svm = [1 if label == 1 else -1 for label in y_train]

        # Parameters for training
        C = 1.0
        learning_rate = 0.01
        iterations = 1000

        print(f"SVM Parameters:")
        print(f"  - C (regularization): {C}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Iterations: {iterations}\n")

        # Create and train SVM model
        model = gdv_vis.SVM()
        model.fit(X_train, y_train_svm, C, learning_rate, iterations, verbose=True)

        # Predictions on train and validation
        y_train_pred_svm = [model.predict(row) for row in X_train]
        y_val_pred_svm   = [model.predict(row) for row in X_val]

        # Convert predictions from {-1,1} back to {0,1} for metrics
        y_train_pred = [1 if p == 1 else 0 for p in y_train_pred_svm]
        y_val_pred   = [1 if p == 1 else 0 for p in y_val_pred_svm]

        # Metrics (binary)
        print("TRAIN SET METRICS (binary):")
        print(f"Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
        print(f"Precision: {precision_score(y_train, y_train_pred, average='binary'):.4f}")
        print(f"Recall:    {recall_score(y_train, y_train_pred, average='binary'):.4f}")
        print(f"F1-score:  {f1_score(y_train, y_train_pred, average='binary'):.4f}\n")
        print("\n")

        print("VALIDATION SET METRICS (binary:")
        print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
        print(f"Precision: {precision_score(y_val, y_val_pred, average='binary'):.4f}")
        print(f"Recall:    {recall_score(y_val, y_val_pred, average='binary'):.4f}")
        print(f"F1-score:  {f1_score(y_val, y_val_pred, average='binary'):.4f}\n")
        print("\n")

        # Visualization
        print("Generating visualizations...")
        metrics_train = {
            'Accuracy': accuracy_score(y_train, y_train_pred),
            'Precision': precision_score(y_train, y_train_pred, average='binary'),
            'Recall': recall_score(y_train, y_train_pred, average='binary'),
            'F1-score': f1_score(y_train, y_train_pred, average='binary')
        }
        metrics_val = {
            'Accuracy': accuracy_score(y_val, y_val_pred),
            'Precision': precision_score(y_val, y_val_pred, average='binary'),
            'Recall': recall_score(y_val, y_val_pred, average='binary'),
            'F1-score': f1_score(y_val, y_val_pred, average='binary')
        }
        classes = [0, 1]
        Visualizer.create_full_report(
            X_train, y_train, y_train_pred,
            X_val, y_val, y_val_pred,
            metrics_train, metrics_val,
            classes=classes,
            save_path="graphs/svm_binary.png"
        )

        # Additional bar plots
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        Visualizer.plot_metrics_bar(metrics_train, title="Train Metrics")
        plt.subplot(1, 2, 2)
        Visualizer.plot_metrics_bar(metrics_val, title="Validation Metrics")
        plt.tight_layout()
        plt.show()
        print("Visualizations saved and displayed.")