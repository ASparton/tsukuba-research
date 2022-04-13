# External libraries imports
import numpy
from tensorflow.keras.datasets.mnist import load_data as load_mnist

# Internal imports
import inputs.preprocessing as ip
import models.cnn as cnn
import models.knn as knn
import models.random_forest as rf

def get_nb_classification_error(model_predictions: numpy.ndarray, expected_output: numpy.ndarray) -> int:
    """Computes and returns the number of classification errors made by the model predictions."""
    
    nb_errors = 0
    for i in range(len(model_predictions)):
        if model_predictions[i] != expected_output[i]:
            nb_errors += 1
    return nb_errors

def main():
    # DATA
    # Get dataset + split train and test data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Prepare data for the CNN model (normalization & reshaping)
    x_cnn_train = ip.get_cnn_prepared_features(x_train)
    x_cnn_test = ip.get_cnn_prepared_features(x_test)
    
    # Prepare data for the skleanrs model (reshaping)
    x_sklearn_train = x_train.reshape(-1, 28 * 28)
    x_sklearn_test = x_test.reshape(-1, 28 * 28)
    
    # MODELS
    # CNN
    cnn_model = cnn.build_model()
    cnn.train_model(cnn_model, x_cnn_train, y_train, x_cnn_test, y_test)
    cnn_accuracy = cnn.get_model_accuracy(cnn_model, x_cnn_test, y_test)
    print(f"\nCNN --> Model accuracy: {cnn_accuracy}")
    
    # kNN
    knn_model = knn.get_trained_model(x_sklearn_train, y_train)
    knn_accuracy = knn_model.score(x_sklearn_test,y_test)
    print(f"kNN --> Model accuracy: {knn_accuracy}")
    
    # Random forest 
    rf_model = rf.get_trained_model(x_sklearn_train, y_train)
    rf_accuracy = rf_model.score(x_sklearn_test, y_test)
    print(f"Random forest --> Model accuracy: {rf_accuracy}")
    
    # PREDICTIONS
    # CNN
    cnn_predictions = cnn.get_predictions(cnn_model, x_cnn_test)
    print(f"\nCNN --> Number of wrong classification: {get_nb_classification_error(cnn_predictions, y_test)} / 10000.")
    
    # kNN
    knn_predictions = knn_model.predict(x_sklearn_test)
    print(f"kNN --> Number of wrong classification: {get_nb_classification_error(knn_predictions, y_test)} / 10000.")
    
    # Random forest
    rf_predictions = rf_model.predict(x_sklearn_test)
    print(f"Random forest --> Number of wrong classification: {get_nb_classification_error(rf_predictions, y_test)} / 10000.")

if __name__ == '__main__':
    main()