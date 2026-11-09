import os 
import joblib


def test_model_path():
    assert os.path.exists("model/model.pkl")

def test_model():
    model = joblib.load("model/model.pkl")
    assert model is not None


    
        