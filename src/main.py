from tensorflow.keras.datasets.mnist import load_data as load_mnist
import matplotlib.pyplot as plt

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    

if __name__ == '__main__':
    main()