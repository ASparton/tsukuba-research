import numpy
from statistics import mode

from inputs.read import get_belgium_traffic_signs
import inputs.preprocessing as ip
import models.cnn as cnn
import models.knn as knn
import models.random_forest as rf
import reliability

def get_system_predictions(models_predictions : list[list[int]]) -> list[int] :
    """Gets the system predictions by voting rule."""
    
    system_predictions = []
    for i in range(len(models_predictions[0])):
        current_predictions = [model_predictions[i] for model_predictions in models_predictions]
        system_predictions.append(mode(current_predictions))
    return system_predictions

def main():
    # DATA
    # Get dataset + split train and test data
    (x_train, y_train), (x_test, y_test) = get_belgium_traffic_signs()
    
    # Prepare data for the CNN model (normalization & reshaping)
    x_cnn_train = ip.normalize_pixels(x_train)
    x_cnn_test = ip.normalize_pixels(x_test)
    
    # MODELS BUILDING & TRAINING
    print("\n-----Models building and training-----")
    # CNN
    cnn_model = cnn.build_mnist_model()
    cnn.train_mnist_model(cnn_model, x_cnn_train, y_train, x_cnn_test, y_test)
    cnn_reliability = cnn.get_model_reliability(cnn_model, x_cnn_test, y_test)
    print(f"\nCNN --> Model reliability: {cnn_reliability}")
    
    # PREDICTIONS
    print("\n-----Models predictions-----")
    # CNN
    cnn_predictions = cnn.get_predictions(cnn_model, x_cnn_test)
    print(f"\nCNN --> Number of wrong classification: {reliability.get_nb_classification_errors(cnn_predictions, y_test)} / {len(y_test)}.")
    
    # SYSTEMS RESULTS
    # print("\n-----System results-----")
    # system_predictions = get_system_predictions([cnn_predictions, knn_predictions, rf_predictions])
    # nb_system_errors = reliability.get_nb_classification_errors(system_predictions, y_test)
    # system_reliability = reliability.get_tmti_reliability(cnn_predictions, knn_predictions, rf_predictions, y_test)
    # print(f"System reliability: {system_reliability}")
    # print(f"Number of wrong classification: {nb_system_errors} / {len(y_test)}.")

if __name__ == '__main__':
    main()