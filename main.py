import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import sklearn.metrics
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import tree
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LearningCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Classifiers.TestRunner import fit_classifier

# Outlier detection settings
OUTLIERS_DETECTORS = ["IsolationForest", "ZScore"]
DETECTOR = "IsolationForest"
REMOVE_OUTLIERS = True
NUMBER_OF_FEATURES = 30

# Function to detect outliers using Isolation Forest
def outlier_detection_isolation_forest(X: np.ndarray, y:np.ndarray, outlier_class: int = -1) -> np.ndarray:
    isolation_forest = IsolationForest(contamination=0.05)
    res = isolation_forest.fit_predict(X=X)

    # Identify outlier indices
    if outlier_class == -1:
        isolation_forest_outlier_indices = np.where(np.array(res == -1))[0]
    else:
        isolation_forest_outlier_indices = np.where(np.array(res == -1) & (y == outlier_class))[0]

    return isolation_forest_outlier_indices

# Function to detect outliers using Z-score method
def outlier_detection_Z_score(X: np.ndarray, y: np.ndarray, outlier_class: int = -1) -> np.ndarray:
    threshold = 3 # Z-score threshold for identifying outliers
    drop_threshold = 100 # Minimum count of outlier dimensions for a sample to be removed

    z_scores = np.abs(stats.zscore(X))
    z_score_count = np.sum(z_scores > threshold, axis=1)
    res = np.where(z_score_count > drop_threshold, [-1]*z_scores.shape[0], 1)

    if outlier_class == -1:
        z_score_outliers_indices = np.where(np.array(res == -1))[0]
    else:
        z_score_outliers_indices = np.where(np.array(res == -1) & (y == outlier_class))[0]

    return z_score_outliers_indices

# Function to remove detected outliers
def remove_outliers(X, y, indices):
    nX = np.delete(X, indices, axis=0)
    ny = np.delete(y, indices, axis=0)

    return nX, ny

# Function to separate data into training and external test sets
def data_separation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_clean = df[df.y.isin((3, 4))]
    df_clean.reset_index(drop=True, inplace=True)

    # Randomly select 20% of each class for external testing
    rng = np.random.default_rng()
    random_index = rng.choice(df_clean[df_clean.y == 3].index, size=int(len(df_clean[df_clean.y == 3]) * 0.2), replace=False)
    random_index = np.concatenate((random_index, rng.choice(df_clean[df_clean.y == 4].index,
                                                            size=int(len(df_clean[df_clean.y == 4]) * 0.2),
                                                            replace=False)))

    external_test_df = df_clean.iloc[random_index]
    df_clean = df_clean.drop(random_index)
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean, external_test_df

# Function to normalize data using StandardScaler
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

# Function to extract time-domain features from EEG data
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

    return features

# Function to preprocess EEG signals with filtering and feature extraction
def strange_function(row):
    fs = 250  # Sampling frequency (Hz)

    # Low-pass filter (FIR filter design)
    kernel = signal.firwin(51, cutoff=30, fs=fs)  # 30 Hz cutoff
    filtered_signal = np.convolve(row[:400], kernel, mode='valid')
    features = time_domain_features(filtered_signal)
    filtered_signal = (filtered_signal - features["mean"]) / np.sqrt(features["variance"])
    features = time_domain_features(filtered_signal)

    row = np.append(filtered_signal, features["norm_zero_crossing"])

    return row


def select_best_tree(model, X, y):
    tree_accuracies = []
    for tree in model.estimators_:
        tree_predictions = tree.predict(X)
        tree_accuracies.append(sklearn.metrics.accuracy_score(y, tree_predictions))

    return model.estimators_[np.argmax(tree_accuracies)]

def train_models(targets, X, y):
    best_models = {}
    for model, params in targets.items():
        best_model, best_params = fit_classifier(model, X, y, param_grid=params)
        best_models[model.__class__.__name__] = best_model

    return best_models

def run_external_test(models, X, y_true):
    results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X)
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        sensitivity = sklearn.metrics.recall_score(y_true, y_pred, pos_label=4)
        specificity = sklearn.metrics.recall_score(y_true, y_pred, pos_label=3)
        matrix = confusion_matrix(y_true, y_pred, normalize='true')
        results[model_name] = {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, "confusion_matrix": matrix}

    return results


# Main function for loading data, preprocessing, training, and evaluation
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

    # Filtered and extended dataset
    extended_x = np.apply_along_axis(strange_function,1, X)

    models = {
        DecisionTreeClassifier() : {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']},
        SVC() : {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'C': [1e-1, 1e-2, 1e-3, 1e-4],
            'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
        },
        RandomForestClassifier() : {
            'n_estimators': [10, 20, 30, 40, 50],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 10],
            'max_features': ['sqrt'],
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        }}

    # Training loop
    best_models = train_models(models, extended_x, y)
    print(best_models)

    # Learning curves
    for model_name, model in best_models.items():
        LearningCurveDisplay.from_estimator(model, extended_x, y, cv=5, scoring='accuracy')
        plt.title(f'Learning Curve {model_name}')
        plt.grid()
        plt.legend()
        plt.show()

    # External test dataset scaling and filtering
    X_ext, y_ext = data_scaling_ext(external_test_df)
    X_ext = np.apply_along_axis(strange_function, 1, X_ext)

    # External test with stats plotting
    models_metrics = run_external_test(best_models, X_ext, y_ext)
    for model_name, metrics in models_metrics.items():
        print(model_name)
        print("Accuracy:", metrics['accuracy'])
        print("Sensitivity:", metrics['sensitivity'])
        print("Specificity:", metrics['specificity'], end='\n\n')

        ConfusionMatrixDisplay(metrics["confusion_matrix"], display_labels=[3,4]).plot()
        plt.title(f"Confusion matrix {model_name}")
        plt.show()

    '''
    fn = [n for n in range(extended_x.shape[1])]
    cn = ["3", "4"]
    best_tree = select_best_tree(best_models["RandomForestClassifier"], extended_x, y)
    plt.figure(figsize=(20,10))
    tree.plot_tree(best_tree, feature_names=fn, class_names=cn, filled=True, rounded=True, fontsize=10)
    plt.title("Best Tree")
    plt.tight_layout()
    plt.show()
    '''

    '''
    fn = [n for n in range(extended_x.shape[1])]
    cn = ["3", "4"]
    best_tree = best_models["DecisionTreeClassifier"]
    plt.figure(figsize=(20, 10))
    tree.plot_tree(best_tree, feature_names=fn, class_names=cn, filled=True, rounded=True, fontsize=10)
    plt.title("Best Tree")
    plt.tight_layout()
    plt.show()
    '''

    '''
    fn = [n for n in range(extended_x.shape[1])]
    cn = ["3", "4"]
    n_classifier = best_models["RandomForestClassifier"].get_params()['n_estimators']
    fig, ax = plt.subplots(n_classifier // 5, 5, figsize=(25, 25), dpi=800)

    for index, decisionTree in enumerate(best_models["RandomForestClassifier"].estimators_):
        tree.plot_tree(decisionTree, feature_names=fn, class_names=cn, filled=True, ax=ax[index // 5][index % 5])
    plt.show()
    '''



if __name__ == '__main__':
    main()
