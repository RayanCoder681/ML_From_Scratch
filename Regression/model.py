def z_score_normalize(X):
    """
    Normalisation Z-Score (Standardisation) : moyenne=0, écart-type=1.
    
    Formule : X_std = (X - μ) / σ
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std
