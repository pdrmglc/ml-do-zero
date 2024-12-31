# %% Importando bibliotecas
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from models.knn import CustomREGKNN
from functions.metrics import plotRegressionMetrics

# %% Carregar o dataset Diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# %% Separando dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% Criando instâncias dos modelos
# Modelo customizado
knn_custom = CustomREGKNN(k=5)
knn_custom.fit(X_train, y_train)
y_pred_custom = knn_custom.predict(X_test)

# Implementação com sklearn
knn_sklearn = KNeighborsRegressor(n_neighbors=5)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)

# %% Avaliação dos modelos
# Função de avaliação personalizada
plotRegressionMetrics(y_test, y_pred_custom, figName="../imgs/KNNREGMetricsCustom", title="Custom")
plotRegressionMetrics(y_test, y_pred_sklearn, figName="../imgs/KNNREGMetricsSklearn", title="Sklearn")
