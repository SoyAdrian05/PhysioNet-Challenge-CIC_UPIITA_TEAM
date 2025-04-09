from data_loader import *
from CNNN1D import * 
from metrics import *

if __name__ == "__main__":
  x_train,y_train, x_val, y_val = data_process()
  model = create_cnn_model(x_train)
  model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

  epochs = 5

  cnn_model_history = model.fit(x_train, y_train, epochs = epochs, batch_size=10, validation_data=(x_val, y_val))
  plot_result("loss", cnn_model_history)
  plot_result("accuracy", cnn_model_history)

  
  y_pred = model.predict([ x_test_Wb, x_test_r, x_test_g, x_test_b])
  pred_labels = np.argmax(y_pred, axis = 1)
  confusion_matrix_metric(y_val, pred_labels)


