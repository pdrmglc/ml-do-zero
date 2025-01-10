# %% Importar bibliotecas
from sklearn.datasets import make_moons, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from models.Boosting import CustomBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb


# %% Gerar dados de exemplo
# Dados com uma separação não linear
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% Treinar o modelo de boosting customizado
boosting_model = CustomBoostingClassifier(
    base_estimator=DecisionTreeClassifier(),  # Árvore de decisão "fraca"
    n_estimators=50,
    learning_rate=0.1
)
boosting_model.fit(X_train, y_train)

# Comparação com uma única árvore de decisão
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# %% Avaliar o modelo
# Boosting
y_pred_train_boosting = boosting_model.predict(X_train)
accuracy_train_boosting = accuracy_score(y_train, y_pred_train_boosting)
print(f"Boosting Model Accuracy Train: {accuracy_train_boosting:.4f}")

y_pred_boosting = boosting_model.predict(X_test)
accuracy_boosting = accuracy_score(y_test, y_pred_boosting)
print(f"Boosting Model Accuracy Test: {accuracy_boosting:.4f}")

# Árvore de decisão simples
y_pred_train_tree = tree_model.predict(X_train)
accuracy_train_tree = accuracy_score(y_train, y_pred_train_tree)
print(f"Decision Tree Accuracy Train: {accuracy_train_tree:.4f}")

y_pred_tree = tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy Test: {accuracy_tree:.4f}")

# %% Testando modelos clássicos de boosting

ada_model = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
ada_model.fit(X_train, y_train)
ada_accuracy = accuracy_score(y_test, ada_model.predict(X_test))
print(f"AdaBoost Accuracy: {ada_accuracy:.4f}")

xgb_model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
# %%
