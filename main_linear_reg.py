# %% Importando bibliotecas

from RegLinear import CustomLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# %% Criando instâncias dos modelos

model = CustomLinearRegression()
sk_model = LinearRegression()

# %% Gerando dados sintéticos

X = np.random.rand(100, 1) *10
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 3

# %% Ajustando os modelos

model.fit(X, y)
sk_model.fit(X, y)

# %% Identificando os coeficientes

print(model.intercept, model.coefficients)
print(sk_model.intercept_, sk_model.coef_)

