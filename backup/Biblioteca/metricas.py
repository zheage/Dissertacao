def calculate_ks(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate KS statistic
    ks_values = tpr - fpr
    
    # Find the maximum KS statistic
    max_ks_index = np.argmax(ks_values)
    
    return ks_values[max_ks_index]