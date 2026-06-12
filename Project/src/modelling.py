import time

from data_loading import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def split(filepath: str) -> tuple:
    df = load_data(filepath)

    # Split the dataframe into the feature matrix and the target vector.
    X = df.drop("Class", axis=1)  # Features
    y = df["Class"]  # Target labels

    # Split into training and test sets.
    # test_size = 0.2 means 20% of the data is held out for testing.
    # random_state = 42 makes the split reproducible.
    # stratify = y keeps the class imbalance roughly the same in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def _fit_model(estimator, X_train, y_train):
    # Measure how long the model takes to train.
    start_train = time.perf_counter()
    model = estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train
    return model, train_time


def DecisionTree(filepath: str = "../data/creditcard.csv"):
    # Train a decision tree classifier on the data.

    X_train, X_test, y_train, y_test = split(filepath)

    # Limit tree depth a bit to reduce overfitting on the imbalanced dataset.
    estimator = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=10,
    )

    model, train_time = _fit_model(estimator, X_train, y_train)

    return model, train_time, X_test, y_test


def LogRegression(filepath: str = "../data/creditcard.csv"):
    # Train a logistic regression model on the data.

    X_train, X_test, y_train, y_test = split(filepath)

    # StandardScaler normalises the features before logistic regression.
    # class_weight="balanced" gives more weight to the rare fraud cases.
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    )

    model, train_time = _fit_model(estimator, X_train, y_train)

    return model, train_time, X_test, y_test


def RandomForest(filepath: str = "../data/creditcard.csv"):
    # Train a random forest classifier on the data.

    X_train, X_test, y_train, y_test = split(filepath)

    # A random forest uses many trees and averages their predictions.
    # max_depth is capped to limit overfitting and runtime.
    estimator = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        max_depth=10,
    )

    model, train_time = _fit_model(estimator, X_train, y_train)

    return model, train_time, X_test, y_test


def SupportVectorMachine(filepath: str = "../data/creditcard.csv"):
    # Train a linear SVM model on the data.

    X_train, X_test, y_train, y_test = split(filepath)

    # SVMs usually benefit from feature scaling.
    # LinearSVC is the scalable linear version of an SVM.
    estimator = make_pipeline(
        StandardScaler(),
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        ),
    )

    model, train_time = _fit_model(estimator, X_train, y_train)

    return model, train_time, X_test, y_test


def MLP(filepath: str = "../data/creditcard.csv"):
    # Train a multilayer perceptron (small neural network) on the data.

    X_train, X_test, y_train, y_test = split(filepath)

    # MLPs also benefit from feature scaling.
    estimator = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100,),
            random_state=42,
            max_iter=500,
        ),
    )

    model, train_time = _fit_model(estimator, X_train, y_train)

    return model, train_time, X_test, y_test


def get_inference_metrics(model, X_test, y_test):
    # Measure inference time and compute the average precision score.
    # For models that support predict_proba(), use the positive-class probability.
    # For models that do not, use decision_function() instead.

    start_inference = time.perf_counter()

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)
        if y_scores.ndim == 2:
            y_scores = y_scores[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        raise ValueError("Model must support predict_proba or decision_function.")

    inference_time = time.perf_counter() - start_inference
    ap_score = average_precision_score(y_test, y_scores)

    return ap_score, inference_time

def get_energy_metrics(modeltype):
	"""
	A function that will find the energy output for training the model,using the functions above) 
	A function that will then use the test data to find the energy needed for inference
	Return both values
	"""


def evaluate_model(model, X_test, y_test,modeltype):
    # Evaluate the trained model using average precision and inference time.
    ap_score, inference_time = get_inference_metrics(model, X_test, y_test)
    return ap_score, inference_time
