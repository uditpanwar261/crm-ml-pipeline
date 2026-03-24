import pandas as pd

def preprocess_data(df):
    # Encode categorical
    df = pd.get_dummies(df, columns=["source"], drop_first=True)

    X = df.drop(["lead_id", "converted"], axis=1)
    y = df["converted"]

    return X, y