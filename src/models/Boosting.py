import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression


class CustomBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=0.1):
        self.base_estimator = base_estimator or LogisticRegression()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        # Inicializar pesos uniformemente para os exemplos
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        self.models = []
        self.model_weights = []

        for _ in range(self.n_estimators):
            # Criar um clone do modelo base para evitar modificações no original
            model = clone(self.base_estimator)
            
            # Treinar o modelo com os pesos atuais
            model.fit(X, y, sample_weight=sample_weights)
            predictions = model.predict(X)
            
            # Calcular erro ponderado
            err = np.sum(sample_weights * (predictions != y)) / np.sum(sample_weights)
            
            # Se o erro for maior que 50%, parar o treinamento
            if err > 0.5:
                break
            
            # Calcular peso do modelo baseado no erro
            model_weight = self.learning_rate * np.log((1 - err + 1e-10) / (err + 1e-10))
            self.models.append(model)
            self.model_weights.append(model_weight)

            # Atualizar os pesos dos exemplos
            sample_weights *= np.exp(-model_weight * (2 * (predictions != y) - 1))
            sample_weights /= np.sum(sample_weights)  # Normalizar os pesos

        return self

    def predict(self, X):
        # Combinar predições ponderadas de todos os modelos
        # model_predictions = np.array([model.predict(X) for model in self.models])
        # weighted_votes = np.dot(self.model_weights, model_predictions)
        # return (weighted_votes > 0).astype(int)
        model_predictions = np.array([model.predict_proba(X)[:, 1] for model in self.models])
        weighted_votes = np.dot(self.model_weights, model_predictions)
        return (weighted_votes > 0.5).astype(int)



