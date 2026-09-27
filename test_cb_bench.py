import os, sys
import catboost as cb

model_path = 'output/overnight_catboost_gpu.cbm'
if os.path.exists(model_path):
    m = cb.CatBoostClassifier()
    m.load_model(model_path)
    print(f"CatBoost GPU Model loaded successfully!")
    print(f"Number of features: {len(m.feature_names_)}")
    print(f"Feature names: {m.feature_names_}")
else:
    print("Model file not found!")
