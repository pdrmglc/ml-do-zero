import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plotMatrix(y_test, y_pred, labels, figName = "ConfusionMatrix", title=""):

  conf_matrix = confusion_matrix(y_test, y_pred)
  conf_matrix_percent = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)

  # Configuração do gráfico
  class_names = labels
  sns.set(font_scale=1.2)
  plt.figure(figsize=(8, 6))
  sns.heatmap(conf_matrix_percent, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
  for i in range(len(class_names)):
      for j in range(len(class_names)):
          if conf_matrix_percent[i, j] > 0.7:
            color = "white"
          else:
            color="black"
          plt.text(j + 0.5, i + 0.5, f"{conf_matrix_percent[i, j]:.2%}\n({conf_matrix[i, j]})", ha="center", va="center", color=color)
  plt.title(title)
  plt.xlabel('Valores Preditos')
  plt.ylabel('Valores Reais')
  plt.savefig(f'{figName}.png', dpi=300)
  plt.show()