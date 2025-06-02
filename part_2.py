import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from part_1 import (
    create_gd,
    create_gk,
    max_degree,
    number_of_connectivity_components,
    size_max_clique,
    size_max_independent_set,
)


# ==========================================================================================================
def extract_multiple_features(samples, n, k_or_d, graph_type):
    features = []
    if graph_type == "stud":
        graph = create_gd(samples, n, k_or_d)
        features.append(max_degree(n, graph))
        features.append(size_max_independent_set(n, graph))
    else:
        graph = create_gd(samples, n, k_or_d)
        features.append(number_of_connectivity_components(graph))
        features.append(size_max_clique(graph))
    return features


def build_classifier(n, k_or_d, dist1, dist2, type1, iterations=50):
    X = []
    y = []

    for i in range(iterations):
        if dist1 == "stud":
            samples1 = np.random.standard_t(df=3, size=n)
        elif dist1 == "lap":
            samples1 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
        elif dist1 == "weib":
            samples1 = np.random.weibull(a=1 / 2, size=n) * 0.31622776601
        elif dist1 == "exp":
            samples1 = np.random.exponential(scale=1, size=n)

        features1 = extract_multiple_features(samples1, n, k_or_d, dist1)
        X.append(features1)
        y.append(0)

        if dist2 == "stud":
            samples2 = np.random.standard_t(df=3, size=n)
        elif dist2 == "lap":
            samples2 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
        elif dist2 == "weib":
            samples2 = np.random.weibull(a=1 / 2, size=n) * 0.31622776601
        elif dist2 == "exp":
            samples2 = np.random.exponential(scale=1, size=n)

        features2 = extract_multiple_features(samples2, n, k_or_d, dist1)
        X.append(features2)
        y.append(1)

    if dist1 == "stud":
        feature_names = ["max_degree", "size_max_independent_set"]
    else:
        feature_names = ["number_of_connectivity_components", "size_max_clique"]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    if type1 == "har_analyse":
        y_pred = clf.predict(X_test)
        for name, importance in zip(feature_names, clf.feature_importances_):
            print(f"{name}: {importance:.4f}")
    return clf, df


def analyze_feature_importance_vs_n(n_range, k_or_d, dist1, dist2):
    importance_results = {}

    for n in n_range:
        clf, df = build_classifier(n, k_or_d, dist1, dist2, "n_analyse", iterations=50)
        importance_results[n] = clf.feature_importances_

    plt.figure(figsize=(12, 6))
    for feature_idx in range(len(clf.feature_importances_)):
        importances = [importance_results[n][feature_idx] for n in n_range]
        plt.plot(n_range, importances, label=f"Признак {feature_idx}")

    plt.xlabel("Размер n")
    plt.ylabel("Важность признака")
    plt.title("Зависимость важности от n")
    plt.legend()
    plt.grid()
    plt.show()


def t_classifier_1(classifier, dist, n=50):
    targets = [1] * n + [0] * n
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    for i in range(n):
        samples = np.random.standard_t(df=3, size=n)
        graph = create_gd(samples, n, dist)
        a = max_degree(n, graph)
        b = size_max_independent_set(n, graph)
        predict = classifier(a, b)
        if predict == targets[i]:
            if targets[i] == 1:
                true_true += 1
            else:
                true_false += 1
        else:
            if targets[i] == 1:
                false_true += 1
            else:
                false_false += 1
    for i in range(n, 2 * n):
        samples = np.random.laplace(loc=0, scale=0.70710678118, size=n)
        graph = create_gd(samples, n, dist)
        a = max_degree(n, graph)
        b = size_max_independent_set(n, graph)
        predict = classifier(a, b)
        if predict == targets[i]:
            if targets[i] == 1:
                true_true += 1
            else:
                true_false += 1
        else:
            if targets[i] == 1:
                false_true += 1
            else:
                false_false += 1
    print("Ошибка первого рода: ", true_false / 2 * n)
    print("Мощность: ", true_true / 2 * n)
    print("Точность: ", (true_true + false_false) / 2 * n)
    return [true_false / 2 * n, true_true / 2 * n, (true_true + false_false) / 2 * n]


def t_classifier_2(classifier, dist, n=50):
    targets = [1] * n + [0] * n
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    for i in range(n):
        samples = np.random.weibull(a=1 / 2, size=n) * 0.31622776601
        graph = create_gd(samples, n, dist)
        a = number_of_connectivity_components(graph)
        b = size_max_clique(graph)
        predict = classifier(a, b)
        if predict == targets[i]:
            if targets[i] == 1:
                true_true += 1
            else:
                true_false += 1
        else:
            if targets[i] == 1:
                false_true += 1
            else:
                false_false += 1
    for i in range(n, 2 * n):
        samples = np.random.exponential(scale=1, size=n)
        graph = create_gd(samples, n, dist)
        a = number_of_connectivity_components(graph)
        b = size_max_clique(graph)
        predict = classifier(a, b)
        if predict == targets[i]:
            if targets[i] == 1:
                true_true += 1
            else:
                true_false += 1
        else:
            if targets[i] == 1:
                false_true += 1
            else:
                false_false += 1
    print("Ошибка первого рода: ", true_false / 2 * n)
    print("Мощность: ", true_true / 2 * n)
    print("Точность: ", (true_true + false_false) / 2 * n)
    return [true_false / 2 * n, true_true / 2 * n, (true_true + false_false) / 2 * n]


def Analyze_of_metric(n_values, k_or_d, dist1, dist2, classifier_name):
    results = []
    classifiers = {
        "Дерево": RandomForestClassifier(n_estimators=100, random_state=42),
        "Логистическая регрессия": LogisticRegression(max_iter=1000, random_state=42),
        "K-ближайших соседей": KNeighborsClassifier(n_neighbors=5),
    }
    selected_clf = classifiers[classifier_name]

    for n in n_values:
        X = []
        y = []
        for i in range(50):
            if dist1 == "stud":
                samples1 = np.random.standard_t(df=3, size=n)
            elif dist1 == "lap":
                samples1 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
            elif dist1 == "weib":
                samples1 = np.random.weibull(a=1 / 2, size=n) * 0.31622776601
            elif dist1 == "exp":
                samples1 = np.random.exponential(scale=1, size=n)

            features1 = extract_multiple_features(samples1, n, k_or_d, dist1)
            X.append(features1)
            y.append(0)

            if dist2 == "stud":
                samples2 = np.random.standard_t(df=3, size=n)
            elif dist2 == "lap":
                samples2 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
            elif dist2 == "weib":
                samples2 = np.random.weibull(a=1 / 2, size=n) * 0.31622776601
            elif dist2 == "exp":
                samples2 = np.random.exponential(scale=1, size=n)

            features2 = extract_multiple_features(samples2, n, k_or_d, dist1)
            X.append(features2)
            y.append(1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        n_metrics = []
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            n_metrics.append(
                {
                    "Классификатор": name,
                    "Точность": acc,
                    "Precision": report["1"]["precision"],
                    "Recall": report["1"]["recall"],
                    "F1-score": report["1"]["f1-score"],
                }
            )

        results.append({"n": n, "Метрики": pd.DataFrame(n_metrics)})

    print("\nРезультаты анализа метрик для различных n:")
    for result in results:
        print(f"\nРазмер выборки n = {result['n']}")
        print(result["Метрики"])
    selected_clf.fit(X, y)
    return selected_clf


def create_classifier_wrapper(clf):
    def wrapper(a, b):
        features = np.array([[a, b]])
        return clf.predict(features)[0]

    return wrapper
