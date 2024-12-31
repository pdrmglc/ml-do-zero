# %% Importando bibliotecas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from models.knn import CustomCLFKNN
from functions.confusionMatrix import plotMatrix

# %%
# Carregar o dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# %%
# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %% Separando dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# %% Criando instâncias dos modelos
knn_custom = CustomCLFKNN(k=5)
knn_custom.fit(X_train, y_train)
y_pred_custom = knn_custom.predict(X_test)

# Implementação com sklearn
knn_sklearn = KNeighborsClassifier(n_neighbors=5)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)

# %% Avaliação dos modelos
labels = ['Setosa','Versicolor','Virginica']

plotMatrix(y_test, y_pred_custom, labels, figName = "../imgs/KNNCLFConfusionMatrixCustom", title="Custom")
plotMatrix(y_test, y_pred_sklearn, labels, figName = "../imgs/KNNCLFConfusionMatrixSklearn", title="Sklearn")
