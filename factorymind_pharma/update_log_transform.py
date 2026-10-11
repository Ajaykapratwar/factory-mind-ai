import sys
import numpy as np

filepath = r'c:\Users\dell\Downloads\factory-mind-ai\factorymind_pharma\services\forecast_svc\ensemble_trainer.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. train_lgbm_expanded
content = content.replace(
    'y_train = train_clean["units_sold"]',
    'y_train = np.log1p(train_clean["units_sold"])'
)
content = content.replace(
    'y_val   = val_clean["units_sold"]',
    'y_val   = np.log1p(val_clean["units_sold"])'
)
content = content.replace(
    'preds = np.maximum(model.predict(X_test), 0)',
    'preds = np.expm1(np.maximum(model.predict(X_test), 0))'
)

# 2. train_per_segment_lgbm
content = content.replace(
    'y_tr  = tr_clean["units_sold"]',
    'y_tr  = np.log1p(tr_clean["units_sold"])'
)
content = content.replace(
    'y_val_s = val_clean["units_sold"] if len(val_clean) > 0 else y_tr.head(10)',
    'y_val_s = np.log1p(val_clean["units_sold"]) if len(val_clean) > 0 else y_tr.head(10)'
)
content = content.replace(
    'te_preds = np.maximum(model.predict(X_te), 0)',
    'te_preds = np.expm1(np.maximum(model.predict(X_te), 0))'
)

# 3. _train_base_models_oof
content = content.replace(
    'y_combined  = combined_clean["units_sold"].values',
    'y_combined  = np.log1p(combined_clean["units_sold"].values)'
)
content = content.replace(
    'lgb_oof[va_mask] = np.maximum(lgb_model.predict(Xva), 0)',
    'lgb_oof[va_mask] = np.expm1(np.maximum(lgb_model.predict(Xva), 0))'
)
content = content.replace(
    'xgb_oof[va_mask] = np.maximum(xgb_model.predict(Xva), 0)',
    'xgb_oof[va_mask] = np.expm1(np.maximum(xgb_model.predict(Xva), 0))'
)
content = content.replace(
    'cb_oof[va_mask] = np.maximum(cb_model.predict(Xva), 0)',
    'cb_oof[va_mask] = np.expm1(np.maximum(cb_model.predict(Xva), 0))'
)
content = content.replace(
    'lgb_test = np.maximum(lgb_final.predict(X_test), 0)',
    'lgb_test = np.expm1(np.maximum(lgb_final.predict(X_test), 0))'
)
content = content.replace(
    'xgb_test = np.maximum(xgb_final.predict(X_test), 0)',
    'xgb_test = np.expm1(np.maximum(xgb_final.predict(X_test), 0))'
)
content = content.replace(
    'cb_test = np.maximum(cb_final.predict(X_test), 0)',
    'cb_test = np.expm1(np.maximum(cb_final.predict(X_test), 0))'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
