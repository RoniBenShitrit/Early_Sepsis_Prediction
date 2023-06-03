import pandas as pd
import os
from tqdm import tqdm
import argparse
import pickle

MODEL_PATH = 'xgboost_5_rows_back.pkl'


def load_data(data_path):
    def stats(df, rows_to_include):
        is_sick = bool(len(df[df.SepsisLabel == 1]))
        if is_sick:
            index = df[df.SepsisLabel == 1].index[0]
            df = df[:index + 1]
        df = df.drop(['SepsisLabel'], axis=1, errors='ignore')
        df = df.tail(rows_to_include + 1).mean(axis=0)
        return df

    file_list = os.listdir(data_path)
    all_patients = []
    for i, file in tqdm(enumerate(file_list)):
        df_all = pd.read_csv(data_path + file, sep='|')
        df_all = stats(df_all, 5)
        all_patients.append(df_all)
    file_list = [file.split(".")[0] for file in file_list]
    return file_list, pd.DataFrame(all_patients)


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def main(file_path):
    model = load_model(MODEL_PATH)
    patients, df = load_data(file_path)
    y_pred = model.predict(df)
    pd.DataFrame({'id': patients, 'prediction': y_pred}).to_csv("prediction.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="path to the file")
    args = parser.parse_args()
    main(args.file_path)
