"""Optional Keras model integration. No diagnosis is guessed without trained weights."""
import os
import numpy as np
MODEL_PATH = "models/plant_disease_model.keras"
LABEL_PATH = "models/disease_labels.txt"
def predict_disease(image):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        return {"ok":False,"message":"Disease detection model is not installed. Add a trained Keras model at models/plant_disease_model.keras and matching class names in models/disease_labels.txt. No diagnosis was made."}
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        arr = np.asarray(image.resize((224,224)), dtype=np.float32)/255.0
        probs = model.predict(np.expand_dims(arr,0), verbose=0)[0]
        with open(LABEL_PATH, encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) != len(probs):
            return {"ok":False,"message":"Model classes do not match disease_labels.txt."}
        i = int(np.argmax(probs))
        return {"ok":True,"label":labels[i],"confidence":float(probs[i]*100)}
    except Exception as e:
        return {"ok":False,"message":f"Model could not run: {e}"}
