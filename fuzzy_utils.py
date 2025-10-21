import numpy as np

# Gaussian membership functions (mean و sigma تیون شده بر اساس thresholds قبلی)
def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean)**2) / (2 * sigma**2))

def membership_low(x):
    return gaussmf(x, 0.3, 0.2)  # peak در 0.3 (بین 0.1-0.5)

def membership_medium(x):
    return gaussmf(x, 0.6, 0.3)  # peak در 0.6 (بین 0.1-1.2)

def membership_high(x):
    return gaussmf(x, 0.9, 0.3)  # peak در 0.9 (از 0.4+)

def fuzzy_lr_scaling(avg_loss):
    """
    If the average loss is low => increase the learning rate (1.2)
    If average loss is medium => keep the learning rate constant (1.0)
    If average loss is high => decrease the learning rate (0.8)
    """
    low_m = membership_low(avg_loss)
    med_m = membership_medium(avg_loss)
    high_m = membership_high(avg_loss)
    denom = low_m + med_m + high_m
    if denom == 0:
        return 1.0
    return (low_m * 1.2 + med_m * 1.0 + high_m * 0.8) / denom

# task_weight_fuzzy حالا در maml.py dynamic می‌شه، اینجا حذف