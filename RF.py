import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

# 读取数据
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')['label']

# 自定义评分函数，包括置信区间计算
def custom_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    auc = roc_auc_score(y_true, y_pred)
    
    # 计算置信区间
    se_ci = proportion_confint(tp, tp + fn, alpha=0.05, method='wilson')
    sp_ci = proportion_confint(tn, tn + fp, alpha=0.05, method='wilson')
    ppv_ci = proportion_confint(tp, tp + fp, alpha=0.05, method='wilson')
    npv_ci = proportion_confint(tn, tn + fn, alpha=0.05, method='wilson')
    
    return (sensitivity, se_ci), (specificity, sp_ci), (ppv, ppv_ci), (npv, npv_ci), auc

def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # 计算95%置信区间
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    return ci_lower, ci_upper

# 设定K折交叉验证参数（使用5折交叉验证）
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 训练和评估RF模型
rf_model = RandomForestClassifier(n_estimators=100)

# 交叉验证预测
y_pred_cv_rf = cross_val_predict(rf_model, X, y, cv=kf)

# 计算指标和置信区间
(sensitivity_rf, se_ci_rf), (specificity_rf, sp_ci_rf), (ppv_rf, ppv_ci_rf), (npv_rf, npv_ci_rf), auc_rf = custom_scorer(y, y_pred_cv_rf)

# 计算AUC的95%置信区间
auc_ci_lower_rf, auc_ci_upper_rf = bootstrap_auc(y, y_pred_cv_rf)

# 输出结果
print(f"RF 5-Fold CV Sensitivity (Se): {sensitivity_rf:.2f} (95% CI: {se_ci_rf[0]:.2f}-{se_ci_rf[1]:.2f})")
print(f"RF 5-Fold CV Specificity (Sp): {specificity_rf:.2f} (95% CI: {sp_ci_rf[0]:.2f}-{sp_ci_rf[1]:.2f})")
print(f"RF 5-Fold CV PPV: {ppv_rf:.2f} (95% CI: {ppv_ci_rf[0]:.2f}-{ppv_ci_rf[1]:.2f})")
print(f"RF 5-Fold CV NPV: {npv_rf:.2f} (95% CI: {npv_ci_rf[0]:.2f}-{npv_ci_rf[1]:.2f})")
print(f"RF 5-Fold CV AUC: {auc_rf:.2f} (95% CI: {auc_ci_lower_rf:.2f}-{auc_ci_upper_rf:.2f})")
