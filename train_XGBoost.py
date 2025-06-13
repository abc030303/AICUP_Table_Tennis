import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, plot_importance
from tqdm import tqdm
import joblib

TRAIN_INFO = "./39_Training_Dataset/train_info.csv"
TEST_INFO = "./39_Test_Dataset/test_info.csv"
TRAIN_DIR = "./39_Training_Dataset/train_data"
TEST_DIR = "./39_Test_Dataset/test_data"
SUBMIT_FILE = "submission_XGBoost.csv"
MODEL_DIR = "./saved_models"

os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = {
    "gender": "binary",
    "hold racket handed": "binary",
    "play years": "multiclass",
    "level": "multiclass"
}

def extract_features(df_dir, info_df):
    all_features = []
    for _, row in tqdm(info_df.iterrows(), total=len(info_df)):
        uid = row["unique_id"]
        fpath = os.path.join(df_dir, f"{uid}.txt")
        raw = pd.read_csv(fpath, header=None, sep=r'\s+', dtype=float, on_bad_lines='skip').dropna()
        features = {}
        features["unique_id"] = uid
        for i, axis in enumerate(["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]):
            data = raw.iloc[:, i]
            features[f"{axis}_mean"] = np.mean(data)
            features[f"{axis}_std"] = np.std(data)
            features[f"{axis}_max"] = np.max(data)
            features[f"{axis}_min"] = np.min(data)
            features[f"{axis}_kurtosis"] = pd.Series(data).kurtosis()
            features[f"{axis}_skew"] = pd.Series(data).skew()
        all_features.append(features)
    return pd.DataFrame(all_features)

def cross_val_auc(X, y, task_type, class_names=None, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if task_type == "binary":
            model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42)
            model.fit(X_tr, y_tr)
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
        else:
            model = XGBClassifier(objective="multi:softprob", num_class=len(class_names), use_label_encoder=False, eval_metric='mlogloss', random_state=42)
            model.fit(X_tr, y_tr)
            y_pred = model.predict_proba(X_val)
            auc = roc_auc_score(y_val, y_pred, multi_class='ovr')

        aucs.append(auc)

    print(f"Cross-Validation AUC ({task_type}): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

def train_and_predict(X_train, y_train, X_test, task_type, class_names=None, model_name="model"):
    if task_type == "binary":
        model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42) #best 300 
        model.fit(X_train, y_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, y_prob_train)
    else:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(class_names),
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_estimators=200, # best 200
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1, # best 0.1
            reg_lambda=1.0,
            random_state=42
        )
                
        model.fit(X_train, y_train)
        y_prob_train = model.predict_proba(X_train)
        auc = roc_auc_score(y_train, y_prob_train, multi_class="ovr")

    print(f"AUC (train): {auc:.4f}")

    # plot_importance(model, max_num_features=20)
    # plt.title(f"Feature Importance: {model_name}")
    # plt.tight_layout()
    # plt.show()

    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.pkl"))

    if task_type == "binary":
        return model.predict_proba(X_test)[:, 1]
    else:
        return model.predict_proba(X_test)

def main():
    train_info = pd.read_csv(TRAIN_INFO)
    test_info = pd.read_csv(TEST_INFO)

    print("Extracting train features...")
    train_feat = extract_features(TRAIN_DIR, train_info)
    print("Extracting test features...")
    test_feat = extract_features(TEST_DIR, test_info)

    submission = pd.DataFrame()
    submission["unique_id"] = test_feat["unique_id"]

    for target, task_type in TARGETS.items():
        print(f"\nTraining: {target}")
        X_train = train_feat.drop(columns=["unique_id"])
        X_test = test_feat.drop(columns=["unique_id"])

        if task_type == "binary":
            y_train = (train_info[target] == 1).astype(int)
            cross_val_auc(X_train, y_train, task_type)
            y_prob = train_and_predict(X_train, y_train, X_test, task_type, model_name=target.replace(" ", "_"))
            submission[target] = np.round(y_prob, 3)

        elif target == "play years":
            y_train = train_info[target]
            class_list = [0, 1, 2]
            cross_val_auc(X_train, y_train, task_type, class_list)
            y_prob = train_and_predict(X_train, y_train, X_test, task_type, class_names=class_list, model_name=target.replace(" ", "_"))
            for i, col in enumerate(["play years_0", "play years_1", "play years_2"]):
                submission[col] = np.round(y_prob[:, i], 3)

        elif target == "level":
            y_train = train_info[target] - 2
            class_list = [0, 1, 2, 3]
            cross_val_auc(X_train, y_train, task_type, class_list)
            y_prob = train_and_predict(X_train, y_train, X_test, task_type, class_names=class_list, model_name=target.replace(" ", "_"))
            for i, col in enumerate(["level_2", "level_3", "level_4", "level_5"]):
                submission[col] = np.round(y_prob[:, i], 3)

    print("Saving submission.csv")
    submission.to_csv(SUBMIT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
