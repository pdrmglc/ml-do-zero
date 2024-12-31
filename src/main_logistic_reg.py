# %% Importando bibliotecas

from models.RegLogistic import CustomLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from functions.confusionMatrix import plotMatrix

# %% Criando instâncias dos modelos

model = CustomLogisticRegression(learning_rate=0.1, max_iter=1000)
sk_model = LogisticRegression()

# %% Baixando dados

# Carrega o dataset
cancer = load_breast_cancer()

# Exibe as features e os targets
X = cancer.data  # Matriz de features
y = cancer.target  # Vetor de target

# %% Separando dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.20)

# %% Ajustando os modelos

model.fit(X, y)
sk_model.fit(X, y)

y_pred_custom = model.predict(X_test)
y_pred_sklearn = sk_model.predict(X_test)

# %% Avaliação dos modelos

labels = ['Tumor maligno', 'Tumor benigno']

plotMatrix(y_test, y_pred_custom, labels, figName = "../imgs/LogisticRegConfusionMatrixCustom", title="Custom")
plotMatrix(y_test, y_pred_sklearn, labels, figName = "../imgs/LogisticRegConfusionMatrixSklearn", title="Sklearn")

# %% Identificando os coeficientes

print("Coeficientes (Custom):", model.coefficients)
print("Intercepto (Custom):", model.intercept)
print("Coeficientes (sklearn):", sk_model.coef_)
print("Intercepto (sklearn):", sk_model.intercept_)

# %% Fazendo previsões

print("Probabilidades previstas (Custom):", model.predict_proba(X))
print("Classes previstas (Custom):", model.predict(X))
print("Probabilidades previstas (sklearn):", sk_model.predict_proba(X))
print("Classes previstas (sklearn):", sk_model.predict(X))