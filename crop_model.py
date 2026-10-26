"""Runnable demonstration model. Replace synthetic data with validated agricultural data."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

CROPS = ["rice","maize","chickpea","banana","cotton","coffee","mango","lentil"]
def train_demo_model():
    rng = np.random.default_rng(42)
    centers = {
        "rice":[80,45,40,25,82,6.2,2200],
        "maize":[75,50,40,24,60,6.3,700],
        "chickpea":[40,65,80,20,20,7.0,650],
        "banana":[100,80,50,27,80,6.2,1100],
        "cotton":[120,45,20,25,55,7.0,800],
        "coffee":[100,20,30,23,60,6.0,1700],
        "mango":[20,20,30,30,50,6.5,900],
        "lentil":[20,65,20,22,65,6.5,500]
    }
    X, y = [], []
    for crop, center in centers.items():
        X.extend(rng.normal(center, [12,12,12,3,12,.5,200], size=(100,7)))
        y.extend([crop]*100)
    return RandomForestClassifier(n_estimators=120, random_state=42).fit(np.array(X), np.array(y))
def recommend(model, features):
    probs = model.predict_proba([features])[0]
    idx = np.argsort(probs)[::-1][:3]
    return [(model.classes_[i], float(probs[i]*100)) for i in idx]
