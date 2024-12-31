# Recriando Algoritmos de Machine Learning

Este repositório tem como objetivo implementar manualmente os principais algoritmos de aprendizado de máquina do zero, sem o uso de bibliotecas como scikit-learn para o core dos algoritmos. O foco é entender os fundamentos matemáticos, lógicos e computacionais que sustentam cada técnica.

## Algoritmos a serem implementados

### Regressão

- Regressão linear ✔
- KNN ✔
- SVM
- Árvore de decisão
- Random Forest
- Gradient Boosting

### Classificação

- Regressão logística ✔
- KNN ✔
- SVM
- Árvore de decisão
- Random Forest
- Gradient Boosting

## Estrutura do repositório
```bash
.
├── README.md
├── imgs
│   ├── KNNCLF.png
│   ├── KNNCLFCLASS.png
│   ├── KNNCLFConfusionMatrixCustom.png
│   ├── KNNCLFConfusionMatrixSklearn.png
│   ├── KNNREG.png
│   ├── KNNREGCLASS.png
│   ├── KNNREGMetricsCustom.png
│   ├── KNNREGMetricsSklearn.png
│   ├── LinearReg.png
│   ├── LogisticReg.png
│   ├── LogisticRegConfusionMatrixCustom.png
│   └── LogisticRegConfusionMatrixSklearn.png
└── src
    ├── functions
    │   ├── __init__.py
    │   ├── confusionMatrix.py
    │   └── metrics.py
    ├── main_knn_clf.py
    ├── main_knn_reg.py
    ├── main_linear_reg.py
    ├── main_logistic_reg.py
    └── models
        ├── RegLinear.py
        ├── RegLogistic.py
        ├── __init__.py
        └── knn.py
```