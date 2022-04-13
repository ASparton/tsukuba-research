from sklearn.neighbors import KNeighborsClassifier

def get_trained_model(features, labels):
    """Builds and trains a knn model with the given features and labels."""
    
    model = KNeighborsClassifier(3, weights='distance')
    model.fit(features, labels)
    return model