'''
A Multi Layer Perceptron model that is trained to recognize HandWritten numbers

Dataset Used: MNIST

Layers structure

Input(28x28) => Hidden1(128,activ = relu) => Hidden2(128, activ = relu) => Output(10, activ = softmax)

No biases used
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
to_train = False


df = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = df.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))
if to_train:
    model.compile(optimizer = "adam" , loss = "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_train,y_train,epochs = 3)

    val_loss,val_accuracy = model.evaluate(x_test,y_test)
    print(val_loss,val_accuracy)

    model.save("MNIST_MODEL")

new_model = tf.keras.models.load_model("MNIST_MODEL")
predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))