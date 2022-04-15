# External libraries imports
import numpy
from tensorflow.keras.datasets.mnist import load_data as load_mnist
from statistics import mode

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

def get_system_predictions(cnn_predictions : list[int], knn_predictions : list[int], rf_predictions : list[int]) -> list[int] :
    """Gets the system predictions by voting system"""
    
    system_predictions = []
    for i in range(len(cnn_predictions)):
        current_predictions = [cnn_predictions[i], knn_predictions[i], rf_predictions[i]]
        system_predictions.append(mode(current_predictions))
    return system_predictions

def get_system_accuracy(nb_errors : int, nb_predictions : int) -> float :
    """Computes the system accuracy"""
    
    return 1 - (nb_errors / nb_predictions)

def main():
    # DATA
    # Get dataset + split train and test data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    
    # UNCOMMENT TO USE NOISY OR ROTATED IMAGES
    # x_train = ip.get_noisy_images(x_train)
    # x_test = ip.get_noisy_images(x_test)
    
    x_train = ip.get_rotated_images(x_train)
    x_test = ip.get_rotated_images(x_test)
    
    # Prepare data for the CNN model (normalization & reshaping)
    x_cnn_train = ip.get_cnn_prepared_features(x_train)
    x_cnn_test = ip.get_cnn_prepared_features(x_test)
    
    # Prepare data for the skleanrs model (reshaping)
    x_sklearn_train = x_train.reshape(-1, 28 * 28)
    x_sklearn_test = x_test.reshape(-1, 28 * 28)
    
    # MODELS
    print("\n-----Models building and training-----")
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
    print("\n-----Models predictions-----")
    # CNN
    cnn_predictions = cnn.get_predictions(cnn_model, x_cnn_test)
    print(f"\nCNN --> Number of wrong classification: {get_nb_classification_error(cnn_predictions, y_test)} / 10000.")
    
    # kNN
    knn_predictions = knn_model.predict(x_sklearn_test)
    print(f"kNN --> Number of wrong classification: {get_nb_classification_error(knn_predictions, y_test)} / 10000.")
    
    # Random forest
    rf_predictions = rf_model.predict(x_sklearn_test)
    print(f"Random forest --> Number of wrong classification: {get_nb_classification_error(rf_predictions, y_test)} / 10000.")
    
    # SYSTEMS RESULTS
    print("\n-----System results-----")
    # TMSI
    tmsi_predictions = get_system_predictions(cnn_predictions, knn_predictions, rf_predictions)
    nb_tmsi_errors = get_nb_classification_error(tmsi_predictions, y_test)
    tmsi_accuracy = get_system_accuracy(nb_tmsi_errors, len(y_test))
    print(f"\nTMSI --> System accuracy: {tmsi_accuracy}")
    print(f"TMSI --> Number of wrong classification: {nb_tmsi_errors} / 10000.")

if __name__ == '__main__':
    main()