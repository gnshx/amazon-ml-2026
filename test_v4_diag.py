
import traceback
try:
    import src.run_robust_production_v4 as v4
    print('Testing Step 3 feature extraction on small batch...')
    # Let us test extract_features directly
    import json
    with open('output/model_metadata.json') as f:
        meta = json.load(f)
    print('Feat count:', len(meta['feat_names']))
    # test dummy extract
    feats = v4.extract_features(
        'apple inc', '123 main st', 'us', False,
        '123 main st', '12345', '123', {'main'}, ['apple'],
        'apple corp', '123 main st', 'us', False,
        '123 main st', '12345', '123', {'main'}, ['apple'],
        's2', 0, 5, {}
    )
    print('Extracted features length:', len(feats))
    import numpy as np
    from catboost import Pool
    arr = np.array([feats], dtype=np.float32)
    pool = Pool(arr)
    score = v4.cb_model.predict(pool)
    print('Score successfully computed:', score)
except Exception:
    traceback.print_exc()

