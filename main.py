import numpy as np

from part_1 import (Analyze_for_k_and_d, Analyze_of_n, Analyze_of_parametrs,
                    find_A_1, find_A_2)
from part_2 import (Analyze_of_metric, analyze_feature_importance_vs_n,
                    build_classifier, create_classifier_wrapper,
                    t_classifier_1, t_classifier_2)

if __name__ == "__main__":

    par_1 = 3
    par_2 = 0.70710678118
    par_3 = 0.31622776601
    par_4 = 1

    input_k = int(input("Введите k: "))
    input_d = float(input("Введите d: "))
    n = 100
    # =========== 1 ======================
    print("Анализ четырех функций по их параметрам")
    param_range = np.linspace(2, 30, 100)
    Analyze_of_parametrs(param_range, n, input_k, "max_degree", "stud")
    Analyze_of_parametrs(param_range, n, input_k, "max_degree", "lap")

    param_range = np.linspace(2, 30, 100)
    Analyze_of_parametrs(param_range, n, input_d, "size_max_independent_set", "stud")
    Analyze_of_parametrs(param_range, n, input_d, "size_max_independent_set", "lap")

    param_range = np.linspace(0.5, 10, 70)
    Analyze_of_parametrs(
        param_range, n, input_k, "number_of_connectivity_components", "weib"
    )
    Analyze_of_parametrs(
        param_range, n, input_k, "number_of_connectivity_components", "exp"
    )

    param_range = np.linspace(0.5, 10, 60)
    Analyze_of_parametrs(param_range, n, input_d, "size_max_clique", "weib")
    Analyze_of_parametrs(param_range, n, input_d, "size_max_clique", "exp")

    # =========== 2 ======================
    print("Анализ четырех функций по k, d и n")
    print("1) max_degree")
    n_range = range(50, 400, 10)
    k_range = range(2, 20)
    Analyze_for_k_and_d(par_1, n, k_range, "stud", "max_degree")
    Analyze_for_k_and_d(par_2, n, k_range, "lap", "max_degree")
    Analyze_of_n(par_1, "stud", n_range, input_k, "max_degree")
    Analyze_of_n(par_2, "lap", n_range, input_k, "max_degree")

    print("2) size_max_independent_set")
    d_range = np.linspace(0.05, 10, 30)
    Analyze_for_k_and_d(par_2, n, d_range, "lap", "size_max_independent_set")
    Analyze_for_k_and_d(par_1, n, d_range, "stud", "size_max_independent_set")
    Analyze_of_n(par_2, "lap", n_range, input_d, "size_max_independent_set")
    Analyze_of_n(par_2, "stud", n_range, input_d, "size_max_independent_set")

    print("3) number_of_connectivity_components")
    n_range = range(50, 100, 2)
    k_range = range(2, 20)
    Analyze_for_k_and_d(par_3, n, k_range, "exp", "number_of_connectivity_components")
    Analyze_for_k_and_d(par_4, n, k_range, "weib", "number_of_connectivity_components")
    Analyze_of_n(par_3, "exp", n_range, input_k, "number_of_connectivity_components")
    Analyze_of_n(par_4, "weib", n_range, input_k, "number_of_connectivity_components")

    print("4) size_max_clique")
    d_range = np.linspace(0.05, 10, 30)
    Analyze_for_k_and_d(par_3, n, d_range, "exp", "size_max_clique")
    Analyze_for_k_and_d(par_4, n, d_range, "weib", "size_max_clique")
    Analyze_of_n(par_3, "exp", n_range, input_d, "size_max_clique")
    Analyze_of_n(par_4, "weib", n_range, input_d, "size_max_clique")

    # =========== 3 ======================
    print("Построение множества А1 и А2")

    n = 300
    k = 5
    d = 0.2
    iterations = 1000

    A1_knn = find_A_1(n, "knn", k, iterations)
    A1_dist = find_A_1(n, "dist", d, iterations)
    A2_knn = find_A_2(n, "knn", k, iterations)
    A2_dist = find_A_2(n, "dist", d, iterations)

    # ============= Часть 2 =================

    # =========== 1 ======================
    print("\n")
    print("Исследование важности характеристик")
    print("\n")
    print("1)Важность признаков у stud и lap")
    clf_knn, df_knn = build_classifier(n, input_d, "stud", "lap", "har_analyse")
    print("\n")
    print("2)Важность признаков у exp и weib")
    clf_dist, df_dist = build_classifier(n, input_d, "weib", "exp", "har_analyse")
    print("\n")
    print("Анализ важности признаков в зависимости от размера выборки:")
    n_range = range(20, 201, 40)
    analyze_feature_importance_vs_n(n_range, input_d, "stud", "lap")
    analyze_feature_importance_vs_n(n_range, input_d, "weib", "exp")
    # =========== 2 ======================
    n_range = [25, 100, 500]
    classifier = Analyze_of_metric(
        n_range, input_d, "stud", "lap", "K-ближайших соседей"
    )
    classifier_2 = Analyze_of_metric(n_range, input_d, "weib", "exp", "Дерево")
    # =========== 3 ======================
    wrapped_classifier = create_classifier_wrapper(classifier)
    t_classifier_1(wrapped_classifier, input_d)
    wrapped_classifier_2 = create_classifier_wrapper(classifier_2)
    t_classifier_2(wrapped_classifier_2, input_d)
