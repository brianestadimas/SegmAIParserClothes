import numpy as np
import torch
from torch import nn

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


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))