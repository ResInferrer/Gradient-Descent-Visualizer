from sklearn.datasets import make_classification


class DataGenerator():
    def dataset_generation(n_samples: int, n_features: int, n_classes: int) -> tuple:
        """
        Generate a synthetic classification dataset.

        Args:
            n_samples (int): Total number of data points to generate.
            n_features (int): Number of features (dimensions) for each sample.
            n_classes (int): Number of classes.

        Returns:
            A pair (X, y) where:
                - X (ndarray of shape (n_samples, n_features)): Feature matrix.
                - y (ndarray of shape (n_samples,)): Target labels (0 or 1).
        """

        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_classes=n_classes
        )

        return X, y