from sklearn.ensemble import RandomForestClassifier

def get_mnist_trained_model(train_features, train_labels):
    """Builds & trains a random forest ML model with the given training features."""
    
    model = RandomForestClassifier(max_features = 'sqrt',
                                   min_samples_split = 4,
                                   random_state = 0)
    model.fit(train_features, train_labels)
    return model