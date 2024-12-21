import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        """
        Inicializa os parâmetros do modelo de regressão logística.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.coefficients = None  # Para armazenar os coeficientes
        self.intercept = None     # Para armazenar o intercepto
    
    def sigmoid(self, z):
        """
        Função sigmoid: mapeia valores reais para o intervalo (0, 1).
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Ajusta o modelo aos dados, otimizando os coeficientes usando gradiente descendente.
        """
        # Adiciona coluna de 1s para o intercepto
        X = np.c_[np.ones(X.shape[0]), X]
        n_samples, n_features = X.shape
        
        # Inicializa os coeficientes com zeros
        self.coefficients = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # Calcula as probabilidades previstas
            linear_model = X @ self.coefficients
            y_pred = self.sigmoid(linear_model)
            
            # Calcula o gradiente
            gradient = (1 / n_samples) * X.T @ (y_pred - y)
            
            # Atualiza os coeficientes
            self.coefficients -= self.learning_rate * gradient
            
            # Critério de convergência
            if np.linalg.norm(gradient, ord=2) < self.tolerance:
                print(f"Convergiu após {iteration} iterações.")
                break
        
        # Separa intercepto dos outros coeficientes
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
    
    def predict_proba(self, X):
        """
        Retorna as probabilidades previstas para cada amostra.
        """
        # Adiciona coluna de 1s para o intercepto
        X = np.c_[np.ones(X.shape[0]), X]
        linear_model = X @ np.r_[self.intercept, self.coefficients]
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Retorna as classes previstas (0 ou 1) com base no threshold.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
