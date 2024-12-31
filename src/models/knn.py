import numpy as np
from collections import Counter

# Função para calcular a distância Euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Implementação do kNN
class CustomCLFKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Calcula a distância para todos os pontos no conjunto de treinamento
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Obtém os índices dos k vizinhos mais próximos
        k_indices = np.argsort(distances)[:self.k]
        
        # Obtém as classes dos k vizinhos mais próximos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Determina a classe majoritária
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
