from sklearn import svm
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import dump


def svm_grid_search(X_train, y_train, c_range, gamma_range, kernel_options):
    # Define the parameter grid
    param_grid = {"C": c_range, "gamma": gamma_range, "kernel": kernel_options}

    # Initialize the SVM model
    svc = svm.SVC(cache_size=10000, decision_function_shape="ovo")

    grid_search = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=2)
    y_train_flattened = y_train.flatten()

    grid_search.fit(X_train, y_train_flattened)
    return grid_search.best_estimator_, grid_search.best_params_


def calculate_accuracy(model, X_test, y_test):
    y_test_flattened = y_test.flatten()
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test_flattened)
    return accuracy


def plot_images_with_predictions(
    images, labels, predictions, class_names, num_rows=2, num_cols=5
):
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plt.imshow(images[i], interpolation="nearest")
        plt.title(
            f"Actual: {class_names[labels[i][0]]}\nPredicted: {class_names[predictions[i]]}"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(class_names, y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot()
    plt.xticks(rotation=45, ha="right")
    plt.show()


def main(
    num_images=10_000,
    c_range=[0.1, 1, 10],
    gamma_range=["scale"],
    kernel_options=["linear", "rbf", "poly", "sigmoid"],
    n_components=0.99,
    save_model=False,
):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocess the data
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[:num_images])
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled[:num_images])
    X_test_pca = pca.transform(X_test_scaled)

    # Use GridSearchCV to find the best hyperparameters
    best_model, best_params = svm_grid_search(
        X_train_pca, y_train[:num_images], c_range, gamma_range, kernel_options
    )
    print(f"Best Parameters: {best_params}")

    # Save model
    if save_model:
        save_model_filename = "svm_cifar10_model.joblib"
        dump(best_model, save_model_filename)

    # Calculate accuracy with the best model
    accuracy = calculate_accuracy(best_model, X_test_pca, y_test)
    print(f"Model Accuracy with PCA on Test Data: {accuracy * 100:.2f}%")

    # Predict on a subset of test data for visualization
    start = random.randint(0, len(y_test) - 10)
    end = start + 9
    predictions = best_model.predict(X_test_pca[start : end + 1])

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Plot images actual and predicted labels
    plot_images_with_predictions(
        X_test.reshape(-1, 32, 32, 3)[start : end + 1],
        y_test[start : end + 1],
        predictions,
        class_names,
    )

    # Confusion matrix
    plot_confusion_matrix(class_names, y_test.flatten(), best_model.predict(X_test_pca))


if __name__ == "__main__":
    main(save_model=True)
