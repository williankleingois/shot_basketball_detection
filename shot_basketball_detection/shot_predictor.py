import joblib

class ShotPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict_shot(self, angles):
        features_scaled = self.scaler.transform([angles])
        prediction = self.model.predict(features_scaled)
        return prediction[0]