# %% Importando bibliotecas

from models.RegLinear import CustomLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# %% Criando instâncias dos modelos

model = CustomLinearRegression()
sk_model = LinearRegression()

# %% Gerando dados sintéticos (coef = 3, intercept = 2)

X = np.random.rand(1000, 1) *10
y = 3 * X.squeeze() + 2 + np.random.randn(1000) * 3

# %% Ajustando os modelos

model.fit(X, y)
sk_model.fit(X, y)

# %% Identificando os coeficientes

print(model.intercept, model.coefficients)
# >>> Intercept: 1.9125981074452572; coeficientes: [3.00583086]

print(sk_model.intercept_, sk_model.coef_)
# >>> Intercept: 1.9125981074452874; coeficientes: [3.00583086]
