from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import os 
import joblib


def train():
    df = pd.read_csv('data/Iris.csv')
    x = df.drop(columns=["Id","Species"])
    y = df['Species']
    xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier()
    model.fit(xtr,ytr)
    pred = model.predict(xte)
    acs = accuracy_score(yte,pred)
    print(f"Accuracy Score {acs}")

    if acs < 0.8:
        raise ValueError("Model is not accuracte")

    joblib.dump(model,"model/model.pkl")

    print("model is saved")

if __name__ == "__main__":
    train()    