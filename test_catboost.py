from catboost import CatBoost, Pool
import numpy as np, json
with open('output/model_metadata.json') as f:
    meta = json.load(f)
feats = meta['feat_names']
print('num feats:', len(feats))
m = CatBoost()
m.load_model('output/catboost_gpu_model.cbm')
dummy = np.zeros((2, len(feats)), dtype=np.float32)
try:
    p = Pool(dummy)
    pred = m.predict(p)
    print('Predict without feat names:', pred)
except Exception as e:
    print('Predict error without names:', e)

