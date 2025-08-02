import numpy as np

# Functions for fuzzy membership
def membership_low(x):
    if x <= 0.1:
        return 1.0
    elif x < 0.5:
        return (0.5 - x) / (0.5 - 0.1)
    else:
        return 0.0

def membership_medium(x):
    if x < 0.1 or x > 1.0:
        return 0.0
    elif x < 0.3:
        return (x - 0.1) / (0.3 - 0.1)
    elif x <= 0.8:
        return 1.0
    else:
        return (1.0 - x) / (1.0 - 0.8)

def membership_high(x):
    if x <= 0.5:
        return 0.0
    elif x < 0.8:
        return (x - 0.5) / (0.8 - 0.5)
    else:
        return 1.0

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

def task_weight_fuzzy(avg_rel):
    """
    This function uses fuzzy logic to determine the weight of a task based on its reliability.
    """
    # Example fuzzy rules for task weight based on average reliability
    if avg_rel < 0.3:
        return 0.8  # higher importance for tasks with lower reliability
    elif avg_rel < 0.6:
        return 0.5  # medium importance for tasks with medium reliability
    else:
        return 0.2  # lower importance for tasks with higher reliability
