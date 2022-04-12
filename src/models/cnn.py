import numpy as np
from tensorflow import keras

def build_model():
    """Build a CNN model with 2 convolution layers & 1 dense layer."""
    
    # Model parameters
    nb_classes = 10
    pixels_matrix_shape = (28, 28, 1)
    kernel_shape = (3, 3)
    pooling_shape = (2, 2)
    
    model = keras.models.Sequential()
    
    # Input layer
    model.add(keras.layers.Input(pixels_matrix_shape))
    
    # First convolution layer
    model.add(keras.layers.Conv2D(8, kernel_shape, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pooling_shape))
    model.add(keras.layers.Dropout(0.2)) # To maximize the number of neurons that will train
    
    # Second convolution layer
    model.add(keras.layers.Conv2D(16, kernel_shape, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pooling_shape))
    model.add(keras.layers.Dropout(0.2)) # To maximize the number of neurons that will train
    
    
    # With padding & pooling, the image size at this stage is around 5x5 which is too small to continue
    model.add(keras.layers.Flatten())
    
    # First dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.25)) # To maximize the number of neurons that will train
    
    # Second dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.25)) # To maximize the number of neurons that will train
    
    # Output layer
    model.add(keras.layers.Dense(nb_classes, activation='softmax'))
    
    # Compilation
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model_to_train, x_train, y_train, x_test, y_test):
    """Train the given model with the data provided."""
    
    model_to_train.fit(x_train, y_train,
                  batch_size = 600,
                  epochs = 30,
                  verbose = 1,
                  validation_data = (x_test, y_test))

def get_predictions(trained_model, x_test):
    """Returns the final predictions with the given CNN trained model over the given test images."""
    
    predictions = trained_model.predict(x_test)
    return np.argmax(predictions, axis = 1)

def get_model_accuracy(trained_model, x_test, y_test):
    """Returns the trained model accuracy depending on the given test dataset given."""
    
    return trained_model.evaluate(x_test, y_test, verbose = 0)[1]