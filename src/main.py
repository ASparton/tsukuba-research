# External libraries imports
import numpy
from tensorflow.keras.datasets.mnist import load_data as load_mnist
from statistics import mode

# Internal imports
import inputs.preprocessing as ip
import models.cnn as cnn
import models.knn as knn
import models.random_forest as rf
import reliability

def get_system_predictions(models_predictions : list[list[int]]) -> list[int] :
    """Gets the system predictions by voting system"""
    
    system_predictions = []
    for i in range(len(models_predictions[0])):
        current_predictions = [model_predictions[i] for model_predictions in models_predictions]
        system_predictions.append(mode(current_predictions))
    return system_predictions

def main():
    # DATA
    # Get dataset + split train and test data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    
    # UNCOMMENT TO USE NOISY OR ROTATED IMAGES
    x_noisy_train = ip.get_noisy_images(x_train)
    x_noisy_test = ip.get_noisy_images(x_test)
    
    x_rotated_train = ip.get_rotated_images(x_train)
    x_rotated_test = ip.get_rotated_images(x_test)
    
    # Prepare data for the CNN model (normalization & reshaping)
    x_cnn_train = ip.get_cnn_prepared_features(x_rotated_train)
    x_cnn_test = ip.get_cnn_prepared_features(x_rotated_test)
    
    # Prepare data for the skleanrs model (reshaping)
    x_sk_train = x_train.reshape(-1, 28 * 28)
    x_sk_test = x_test.reshape(-1, 28 * 28)
    x_sk_noisy_train = x_noisy_train.reshape(-1, 28 * 28)
    x_sk_noisy_test = x_noisy_test.reshape(-1, 28 * 28)
    # x_sk_rotated_test = x_rotated_test.reshape(-1, 28 * 28)
    # x_sk_rotated_train = x_rotated_train.reshape(-1, 28 * 28)
    
    # MODELS
    print("\n-----Models building and training-----")
    # CNN
    cnn_model = cnn.build_model()
    cnn.train_model(cnn_model, x_cnn_train, y_train, x_cnn_test, y_test)
    cnn_reliability = cnn.get_model_reliability(cnn_model, x_cnn_test, y_test)
    print(f"\nCNN --> Model reliability: {cnn_reliability}")
    
    # kNN
    knn_model = knn.get_trained_model(x_sk_noisy_train, y_train)
    knn_reliability = knn_model.score(x_sk_noisy_test, y_test)
    print(f"kNN --> Model reliability: {knn_reliability}")
    
    # Random forest 
    rf_model = rf.get_trained_model(x_sk_train, y_train)
    rf_reliability = rf_model.score(x_sk_test, y_test)
    print(f"Random forest --> Model reliability: {rf_reliability}")
    
    # PREDICTIONS
    print("\n-----Models predictions-----")
    # CNN
    cnn_predictions = cnn.get_predictions(cnn_model, x_cnn_test)
    print(f"\nCNN --> Number of wrong classification: {reliability.get_nb_classification_errors(cnn_predictions, y_test)} / 10000.")
    
    # kNN
    knn_predictions = knn_model.predict(x_sk_noisy_test)
    print(f"kNN --> Number of wrong classification: {reliability.get_nb_classification_errors(knn_predictions, y_test)} / 10000.")
    
    # Random forest
    rf_predictions = rf_model.predict(x_sk_test)
    print(f"Random forest --> Number of wrong classification: {reliability.get_nb_classification_errors(rf_predictions, y_test)} / 10000.")
    
    # SYSTEMS RESULTS
    print("\n-----System results-----")
    system_predictions = get_system_predictions([cnn_predictions, knn_predictions, rf_predictions])
    nb_system_errors = reliability.get_nb_classification_errors(system_predictions, y_test)
    system_reliability = reliability.get_tmti_reliability(cnn_predictions, knn_predictions, rf_predictions, y_test)
    print(f"System reliability: {system_reliability}")
    print(f"Number of wrong classification: {nb_system_errors} / 10000.")

if __name__ == '__main__':
    main()