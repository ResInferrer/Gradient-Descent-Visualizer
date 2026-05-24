from sklearn.datasets import make_classification


class DataGenerator():
    @staticmethod
    def dataset_generation(n_samples: int, n_features: int, n_classes: int):
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

        n_clusters_per_class = 1
        min_informative = max(2, (n_classes * n_clusters_per_class).bit_length())
        n_informative = min(n_features, min_informative + 2)  # no more n_features

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=2.0,     
            flip_y=0.02,         
            random_state=42
        )

        return X, y