import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle


def preprocess_data(data: pd.DataFrame):
    dataset = data.drop(
        [
            col
            for col in data.columns
            if col.startswith("world")
            or col.startswith("Unnamed")
            or col == "handedness.score"
        ],
        axis=1,
    )
    print(dataset.columns.values)

    hand_column = dataset[["handedness.label"]]

    encoder = OneHotEncoder(sparse_output=False)
    transformed_hand = encoder.fit_transform(hand_column)

    dataset[encoder.categories_[0]] = transformed_hand

    print(dataset.columns.values)

    X = dataset.drop(["handedness.label"], axis=1)
    std = StandardScaler()
    X = std.fit_transform(X)

    model = pickle.load(open("./model_lr.pkl", "rb"))
    predictions = model.predict(X)
    print(predictions)
    return predictions


def perform_processing(data: pd.DataFrame) -> pd.DataFrame:
    # NOTE(MF): sample code
    # preprocessed_data = preprocess_data(data)
    # models = load_models()  # or load one model
    # please note, that the predicted data should be a proper pd.DataFrame with column names
    # predicted_data = predict(models, preprocessed_data)
    # return predicted_data

    predictions = preprocess_data(data)

    # for the simplest approach generate a random DataFrame with proper column names and size
    random_results = np.random.choice(
        [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
        ],
        data.shape[0],
    )
    print(f"{random_results=}")

    predicted_data = pd.DataFrame(predictions, columns=["letter"])

    print(f"{predicted_data=}")

    return predicted_data
