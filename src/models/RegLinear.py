import numpy as np

class CustomLinearRegression:
    def __init__(self):
        """
        Inicializa os atributos da classe.
        """
        self.coefficients = None  # Armazena os coeficientes beta
        self.intercept = None     # Armazena o intercept (beta_0)
        self.fitted = False       # Flag para indicar se o modelo foi ajustado

    def fit(self, X, y):
        """
        Ajusta o modelo aos dados, calculando os coeficientes beta.
        
        Args:
            X (numpy.ndarray): Matriz de features (n x m).
            y (numpy.ndarray): Vetor target (n x 1).
        """
        # Adiciona a coluna de 1s para o intercept
        X = np.c_[np.ones(X.shape[0]), X] 
        
        # Fórmula fechada: beta = (X^T X)^(-1) X^T y
        X_transpose = X.T
        XTX = X_transpose @ X
        XTX_inv = np.linalg.inv(XTX)
        XTy = X_transpose @ y
        beta = XTX_inv @ XTy
        
        # Armazena os coeficientes e o intercept
        self.intercept = beta[0]          # Primeiro coeficiente é o intercept
        self.coefficients = beta[1:]      # Restante são os pesos das features
        self.fitted = True                # Marca como ajustado

    def predict(self, X):
        """
        Faz previsões usando o modelo ajustado.
        
        Args:
            X (numpy.ndarray): Matriz de features (n x m).
        
        Returns:
            numpy.ndarray: Vetor de previsões (n x 1).
        """
        if not self.fitted:
            raise Exception("O modelo não foi ajustado. Execute `fit` primeiro.")
        
        # Adiciona a coluna de 1s para o intercept
        X = np.c_[np.ones(X.shape[0]), X]  # Torna X em [1, x1, x2, ..., xm]
        
        # Previsões: y_hat = X @ beta
        return X @ np.r_[self.intercept, self.coefficients]

    def score(self, X, y):
        """
        Calcula o coeficiente de determinação (R^2) para avaliar o modelo.
        
        Args:
            X (numpy.ndarray): Matriz de features (n x m).
            y (numpy.ndarray): Vetor target (n x 1).
        
        Returns:
            float: O valor de R^2.
        """
        y_pred = self.predict(X)
        total_variance = np.sum((y - np.mean(y))**2)
        residual_variance = np.sum((y - y_pred)**2)
        r2 = 1 - (residual_variance / total_variance)
        return r2
