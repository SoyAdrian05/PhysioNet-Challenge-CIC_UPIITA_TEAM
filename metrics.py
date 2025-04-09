import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

def plot_result(item, model):
    plt.plot(model.history[item], label=item)
    plt.plot(model.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

def confusion_matrix_metric(y_test, pred_labels): 
  
  cm = confusion_matrix(y_test, pred_labels)
  cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
  cm_display.plot()
  plt.show()
  
  
  fp = (cm.sum(axis=0) - np.diag(cm)).astype(float)
  fn = (cm.sum(axis=1) - np.diag(cm)).astype(float)
  tp = (np.diag(cm)).astype(float)
  tn = (cm.sum() - (fp + fn + tp)).astype(float)
  
  recall = tp / (tp + fn)
  specificity = tn / (tn + fp)
  precision = tp / (tp + fp)
  
  #F1
  f1 = (2 * precision*recall) / (precision + recall)
  f1_score(y_test, pred_labels, average='micro') 
  f1_score(y_test, pred_labels, average=None)
  #bar plot
  
  # set width of bar
  barWidth = 0.18
  fig = plt.subplots(figsize =(8, 7),dpi=350)
   
  # Set position of bar on X axis
  br1 = np.arange(len(precision))
  br2 = [x + barWidth for x in br1]
  br3 = [x + barWidth for x in br2]
  br4 = [x + barWidth for x in br3]
  # Make the plot
  plt.bar(br1, precision, color ='#46185F', width = barWidth,
          edgecolor ='grey', label ='Precision')
  plt.bar(br2, specificity, color ='#1B6B93', width = barWidth,
          edgecolor ='grey', label ='Specificity')
  plt.bar(br3, recall, color ='#82CD47', width = barWidth,
          edgecolor ='grey', label ='Recall')
  plt.bar(br4, f1, color ='#FCE22A', width = barWidth,
          edgecolor ='grey', label ='F1_score') 
  # Adding Xticks
  plt.xlabel('Class', fontweight ='bold', fontsize = 12)
  plt.ylabel('Percentage', fontweight ='bold', fontsize = 12)
  plt.xticks([r + barWidth for r in range(len(precision))],
          ['ap','am','b','cat','deer','dog','frog','horse','ship','truck'])
  plt.ylim(0,1)
  plt.legend(loc="lower right")
  plt.show()
