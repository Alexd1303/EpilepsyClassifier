import random
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.signal as signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, LearningCurveDisplay, learning_curve, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score

from Classifiers.TestRunner import fit_decision_tree, fit_svm, fit_classifier

OUTLIERS_DETECTORS = ["IsolationForest", "ZScore"]
DETECTOR = "IsolationForest"
REMOVE_OUTLIERS = True
NUMBER_OF_FEATURES = 30

def outlier_detection_isolation_forest(X: np.ndarray, y:np.ndarray, outlier_class: int = -1) -> np.ndarray:
    isolationForest = IsolationForest(contamination=0.05)
    res = isolationForest.fit_predict(X=X)
    if outlier_class == -1:
        isolationForest_outlier_indices = np.where(np.array(res == -1))[0]
    else:
        isolationForest_outlier_indices = np.where(np.array(res == -1) & (y == outlier_class))[0]

    return isolationForest_outlier_indices

def outlier_detection_Z_score(X: np.ndarray, y: np.ndarray, outlier_class: int = -1) -> np.ndarray:
    threshold = 3
    drop_threshold = 100

    z_scores = np.abs(stats.zscore(X))
    z_score_count = np.sum(z_scores > threshold, axis=1)
    res = np.where(z_score_count > drop_threshold, [-1]*z_scores.shape[0], 1)

    if outlier_class == -1:
        z_score_outliers_indices = np.where(np.array(res == -1))[0]
    else:
        z_score_outliers_indices = np.where(np.array(res == -1) & (y == outlier_class))[0]

    return z_score_outliers_indices

def remove_outliers(X, y, indices):
    nX = np.delete(X, indices, axis=0)
    ny = np.delete(y, indices, axis=0)

    return nX, ny

def data_separation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_clean = df[df.y.isin((3, 4))]
    df_clean.reset_index(drop=True, inplace=True)

    rng = np.random.default_rng()
    random_index = rng.choice(df_clean[df_clean.y == 3].index, size=int(len(df_clean[df_clean.y == 3]) * 0.2), replace=False)
    random_index = np.concatenate((random_index, rng.choice(df_clean[df_clean.y == 4].index,
                                                            size=int(len(df_clean[df_clean.y == 4]) * 0.2),
                                                            replace=False)))

    external_test_df = df_clean.iloc[random_index]
    df_clean = df_clean.drop(random_index)
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean, external_test_df

def data_scaling(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler3 = StandardScaler()
    X_3 = scaler3.fit_transform(df[df.y == 3].iloc[:, 1:].to_numpy())

    scaler4 = StandardScaler()
    X_4 = scaler4.fit_transform(df[df.y == 4].iloc[:, 1:].to_numpy())

    X_3[:, -1] = 3
    X_4[:, -1] = 4

    plt.figure(figsize=(20, 5))
    plt.plot(X_4[0, :-1], label="Patient 2 Class 4")
    plt.plot(X_3[0, :-1], label="Patient 1 Class 3")
    plt.legend()
    plt.show()

    normalized_df = np.concatenate([X_3, X_4], axis=0)
    np.random.shuffle(normalized_df)
    X = normalized_df[:, :-1]
    y = normalized_df[:, -1]

    return X, y

def data_scaling_ext(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df.iloc[:, 1:-1])
    y = df.iloc[:, -1]

    return X, y


def time_domain_features(eeg_signal):
    features = {
        "mean": np.mean(eeg_signal),
        "variance": np.var(eeg_signal),
        "skewness": skew(eeg_signal),
        "kurtosis": kurtosis(eeg_signal),
        "rms": np.sqrt(np.mean(eeg_signal**2)),
        "zero_crossings": np.sum(np.diff(np.sign(eeg_signal)) != 0),
    }

    features["norm_zero_crossing"] = (features["zero_crossings"] - features["mean"]) / (features["rms"] + 1e-6)
    #features["norm_zero_crossing"] = (features["zero_crossings"] - features["mean"]) / np.sqrt(features["variance"])
    #features["norm_zero_crossing"] = (features["zero_crossings"] - features["mean"]) / (shannon_entropy(eeg_signal) + 1e-6)

    return features

def strange_function(row):
    fs = 250  # Sampling frequency (Hz)

    # Low-pass filter (FIR filter design)
    kernel = signal.firwin(51, cutoff=30, fs=fs)  # 30 Hz cutoff
    filtered_signal = np.convolve(row[:400], kernel, mode='valid')

    features = time_domain_features(filtered_signal)

    filtered_signal = (filtered_signal - features["mean"]) / np.sqrt(features["variance"])

    features = time_domain_features(filtered_signal)

    row = np.append(filtered_signal, features["norm_zero_crossing"])
    #print(features["norm_zero_crossing"])

    return row


def perform_external_test(X_ext, y_ext):
    pass


def main():
    df = pd.read_csv('EEG-data - EEG-data.csv')
    df.info()

    df_clean, external_test_df = data_separation(df)
    X, y = data_scaling(df_clean)
    outliers_iso_forest = outlier_detection_isolation_forest(X, y)
    outliers_z_score = outlier_detection_Z_score(X, y)

    print("outliers isolation forest:", outliers_iso_forest)
    print("outliers z score:", outliers_z_score)

    if REMOVE_OUTLIERS:
        X, y = remove_outliers(X, y, outliers_iso_forest if DETECTOR == "IsolationForest" else outliers_z_score)

    extended_x = np.apply_along_axis(strange_function,1, X)
    print(extended_x.shape)
    pd.DataFrame(extended_x).iloc[0, :-1].plot()
    plt.show()

    models = {
        DecisionTreeClassifier() : {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']},
        SVC() : {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'C': [1e-1, 1e-2, 1e-3, 1e-4],
            'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
        }}

    for model, params in models.items():
        best_model, best_params = fit_classifier(model, extended_x, y, param_grid=params)
        print(best_params)
        print(best_model.score(extended_x, y))

        X_ext, y_ext = data_scaling_ext(external_test_df)
        X_ext = np.apply_along_axis(strange_function, 1, X_ext)
        y_pred = best_model.predict(X_ext)
        ConfusionMatrixDisplay(confusion_matrix(y_ext, y_pred, normalize='true')).plot()
        plt.show()

        print("Accuracy:", sklearn.metrics.accuracy_score(y_ext, y_pred))
        print("Sensitivity", sklearn.metrics.recall_score(y_ext, y_pred, pos_label=4))
        print("Specificity", sklearn.metrics.recall_score(y_ext, y_pred, pos_label=3))

if __name__ == '__main__':
    main()
