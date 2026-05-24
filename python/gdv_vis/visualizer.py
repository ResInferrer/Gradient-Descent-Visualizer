import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.patches import Patch

class Visualizer:
    @staticmethod
    def plot_pca_projection(X, y, title="PCA Projection", ax=None, alpha=0.7, s=30):
        """
        Reduce data to 2D using PCA and plot with class colors.
        
        Args:
            X: feature matrix (n_samples, n_features)
            y: true labels (n_samples,)
            title: plot title
            ax: matplotlib axis (creates new if None)
            alpha: point transparency
            s: point size
        Returns:
            ax: matplotlib axis
        """

        # PCA to 2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique classes and a colormap
        classes = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        
        # Scatter each class
        for cls, color in zip(classes, colors):
            mask = (y == cls)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[color], label=f'Class {cls}',
                      alpha=alpha, s=s, edgecolors='w', linewidth=0.5)
        
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return ax
    
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes=None, title="Confusion Matrix",
                              normalize=True, cmap='Blues', ax=None):
        """
        Plot a beautiful confusion matrix.
        
        Args:
            y_true: ground truth labels
            y_pred: predicted labels
            classes: list of class names (defaults to sorted unique)
            title: plot title
            normalize: if True, show percentages instead of counts
            cmap: colormap
            ax: matplotlib axis
        Returns:
            ax: matplotlib axis
        """

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if classes is None:
            classes = sorted(np.unique(y_true))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title += " (normalized)"
        else:
            fmt = 'd'
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot using seaborn for better aesthetics
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, cbar=True, square=True,
                    annot_kws={'size': 10})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return ax
    

    @staticmethod
    def plot_metrics_bar(metrics_dict, title="Model Performance Metrics", ax=None):
        """
        Plot bar chart of evaluation metrics.
        
        Args:
            metrics_dict: dict with keys 'Accuracy', 'Precision', 'Recall', 'F1-score'
            title: plot title
            ax: matplotlib axis
        Returns:
            ax: matplotlib axis
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(metrics_names)))
        bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        return ax
    

    @staticmethod
    def plot_predictions_vs_true(X, y_true, y_pred, title="Predictions vs True Labels (PCA space)",
                                 ax=None, alpha=0.7, s=30):
        """
        Show two side-by-side PCA plots: true labels and predicted labels.
        """

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        else:
            axes = ax
        
        # True labels plot
        Visualizer.plot_pca_projection(X, y_true, title="True Labels",
                                       ax=axes[0], alpha=alpha, s=s)
        # Predicted labels plot
        Visualizer.plot_pca_projection(X, y_pred, title="Predicted Labels",
                                       ax=axes[1], alpha=alpha, s=s)
        
        return axes
    

    @staticmethod
    def create_full_report(X_train, y_train, y_train_pred, X_val, y_val, y_val_pred,
                           metrics_train, metrics_val, classes=None, save_path=None):
        """
        Create a comprehensive report with multiple subplots.
        
        Args:
            X_train, X_val: feature matrices
            y_train, y_val: true labels
            y_train_pred, y_val_pred: predicted labels
            metrics_train, metrics_val: dicts of metrics
            classes: list of class names
            save_path: if provided, save the figure
        Returns:
            fig: matplotlib figure
        """

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Train PCA projection
        ax1 = fig.add_subplot(gs[0, 0])
        Visualizer.plot_pca_projection(X_train, y_train, title="Training Data (True Labels)",
                                       ax=ax1, alpha=0.6, s=25)
        
        # Validation PCA projection
        ax2 = fig.add_subplot(gs[0, 1])
        Visualizer.plot_pca_projection(X_val, y_val, title="Validation Data (True Labels)",
                                       ax=ax2, alpha=0.6, s=25)
        
        # Confusion matrix (validation)
        ax3 = fig.add_subplot(gs[0, 2])
        Visualizer.plot_confusion_matrix(y_val, y_val_pred, classes=classes,
                                         title="Confusion Matrix (Validation)",
                                         normalize=True, ax=ax3)
        
        # Metrics comparison (train vs val)
        ax4 = fig.add_subplot(gs[1, :])
        x = np.arange(len(metrics_train))
        width = 0.35
        names = list(metrics_train.keys())
        train_vals = list(metrics_train.values())
        val_vals = list(metrics_val.values())
        
        bars1 = ax4.bar(x - width/2, train_vals, width, label='Train', color='#2E86AB', edgecolor='black')
        bars2 = ax4.bar(x + width/2, val_vals, width, label='Validation', color="#C22F2F", edgecolor='black')
        
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Performance Comparison: Train vs Validation', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names)
        ax4.legend(loc='lower right')
        ax4.set_ylim(0, 1.05)
        ax4.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle("Gradient Descent Visualizer - Logistic Regression Report",
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig