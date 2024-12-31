import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def plotRegressionMetrics(y_true, y_pred, figName, title):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Preditos")
    plt.title(f"{title} (RMSE: {rmse:.2f}, R²: {r2:.2f})")
    plt.savefig(figName)
    plt.show()
