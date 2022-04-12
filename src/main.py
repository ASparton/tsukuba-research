import numpy
from tensorflow.keras.datasets.mnist import load_data as load_mnist

import inputs.preprocessing as ip
import models.cnn as cnn

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
    
    # MODELS
    # Build & train CNN model
    cnn_model = cnn.build_model()
    cnn.train_model(cnn_model, x_cnn_train, y_train, x_cnn_test, y_test)
    print(f"CNN --> Model accuracy: {cnn.get_model_accuracy(cnn_model, x_cnn_test, y_test)}")
    
    # PREDICTIONS
    cnn_predictions = cnn.get_predictions(cnn_model, x_cnn_test)
    print(f"CNN --> Number of wrong classification: {get_nb_classification_error(cnn_predictions, y_test)} / 10000.")

if __name__ == '__main__':
    main()