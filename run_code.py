
from data_loader import *
from CNN1D import * 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import numpy as np


if __name__ == "__main__":
  x_train,y_train, x_val, y_val = data_process()
  x_train_split = [x_train[:, i, :].reshape(-1, 136, 1) for i in range(12)]
  x_val_split = [x_val[:, i, :].reshape(-1, 136, 1) for i in range(12)]
  model = create_cnn_model(x_train)
  
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
  y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=2)

  epochs = 100

  cnn_model_history = model.fit(x_train_split, y_train_onehot, epochs = epochs, batch_size=32, validation_data=(x_val_split, y_val_onehot))
  
  y_pred_prob = model.predict(x_val_split)


  y_pred = np.argmax(y_pred_prob, axis=1)
  cm = confusion_matrix(np.argmax(y_val_onehot, axis=1), y_pred)

   
  fig, ax = plt.subplots(figsize=(6, 5))
  cax = ax.matshow(cm, cmap='Blues')
    
    
  for (i, j), value in np.ndenumerate(cm):
     ax.text(j, i, str(value), ha='center', va='center', color='black')
    
   
  ax.set_xlabel('Predicción')
  ax.set_ylabel('Real')
  ax.set_title('Matriz de Confusión')
    
    
  ax.set_xticks([0, 1])
  ax.set_yticks([0, 1])
  ax.set_xticklabels(['Clase 0', 'Clase 1'])
  ax.set_yticklabels(['Clase 0', 'Clase 1'])
    
  plt.show()
