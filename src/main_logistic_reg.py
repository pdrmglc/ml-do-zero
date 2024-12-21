# %% Importando bibliotecas

from models.RegLogistic import CustomLogisticRegression
from sklearn.linear_model import LogisticRegression
import numpy as np

# %% Criando instâncias dos modelos

model = CustomLogisticRegression(learning_rate=0.1, max_iter=1000)
sk_model = LogisticRegression()

# %% Gerando dados simples

X = np.array([[2], [3], [5], [7], [8]])
y = np.array([0, 0, 1, 1, 1])

# %% Ajustando os modelos

model.fit(X, y)
sk_model.fit(X, y)

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