import numpy as np

def relative_root_mean_squared_error(true, pred):
    
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num / den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

def r_score(y_true, y_pred):
    """Compute the correlation coefficient."""
    r_matrix = np.corrcoef(y_true, y_pred)
    r = r_matrix[0, 1]  # Extract the correlation coefficient
    return r