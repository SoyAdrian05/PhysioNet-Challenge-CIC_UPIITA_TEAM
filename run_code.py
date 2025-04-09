from data_loader import *
from CNNN1D import * 

def train_model(model)

if __name__ == "__main__":
  x_train,y_train, x_val, y_val = data_process()
  model = create_cnn_model(x_train)
  model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

  epochs = epochs

  model.history = Convolutional1DNet.fit(x_train, y_train, epochs = epochs, batch_size=10, validation_data=(x_val, y_val))


