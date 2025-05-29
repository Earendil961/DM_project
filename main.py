import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve

def create_gd(x2, n, d):
    gd = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and abs(x2[i] - x2[j]) <= d:
                gd[i][j] = True
                gd[j][i] = True
    return gd

def create_gk(x, n, k):
    n_neighbors = NearestNeighbors(n_neighbors=k).fit(x)
    distances, indices = n_neighbors.kneighbors(x)
    gk = np.zeros((n, n), dtype=bool)
    neighbors_sets = [set() for i in range(n)]
    for i in range(n):
        neighbors = indices[i][1:k]
        neighbors_sets[i].update(neighbors)
    for i in range(n):
        for j in neighbors_sets[i]:
            if i in neighbors_sets[j]:
                gk[i][j] = True
                gk[j][i] = True
    return gk

def size_max_independent_set(n, gd):
    arr = {}
    for i in range(n):
      arr[i] = []
    for i in range(n):
      for j in range(n):
        if gd[i][j]==1:
          arr[i].append(j)
          arr[j].append(i)
    ans = set()
    all_ver = set(arr.keys())
    
    while all_ver:
        v = min(all_ver, key=lambda x: len(arr[x]))
        ans.add(v)
        all_ver.remove(v)
        all_ver -= set(arr[v])
    
    return len(ans)

def max_degree(n, gk):
    arr = np.zeros(n)
    for i in range(n):
      for j in range(n):
        if gk[i][j]==1:
          arr[i]= arr[i] + 1
    m =0
    for i in range(n):
      if m<arr[i]:
        m = arr[i]
    return m

def size_max_clique(adj_matrix):
    n = len(adj_matrix)
    max_clique = 0
    for i in range(n):
        right = i
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                right = j
            else:
                break
        current_clique = right - i + 1
        if current_clique > max_clique:
            max_clique = current_clique
    return max_clique

def number_of_connectivity_components(data):
        n = len(data)
        visited = [False] * n
        count = 0

        def dfs(i):
            visited[i] = True
            for j in range(n):
                if data[i][j] == 1 and not visited[j]:
                    dfs(j)
        for i in range(n):
            if not visited[i]:
                count += 1
                dfs(i)
        return count


def Analyze_of_parametrs(param_range, n, input_k_or_d, type_analyze, type_func, iterations=10):
    t_values = []
    t_val = -1
    samples = []
    for param in param_range:
        sum_val = 0
        for i in range(iterations):
            if type_func == 'stud':
                samples = np.random.standard_t(df=param, size=n)
                samples.sort()
            elif type_func == 'lap':
                samples = np.random.laplace(loc=0, scale=param, size=n)
                samples.sort()
            elif type_func == 'weib':
                samples = np.random.weibull(a=1/2, size=n) * param
                samples.sort()
            elif type_func == 'exp':
                samples = np.random.exponential(scale=1/param, size=n)
                samples.sort()
            if type_analyze == 'max_degree':
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
                t_val = max_degree(n, graph)
            elif type_analyze == 'size_max_independent_set':
                graph = create_gd(samples, n, input_k_or_d)
                t_val = size_max_independent_set(n, graph)
            elif type_analyze == 'number_of_connectivity_components':
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
                t_val = number_of_connectivity_components(graph)
            elif type_analyze == 'size_max_clique':
                graph = create_gd(samples, n, input_k_or_d)
                t_val = size_max_clique(graph)
            sum_val += t_val
        t_values.append(sum_val / iterations)
    plt.figure()
    plt.plot(param_range, t_values)
    plt.xlabel('Parameter value, ' + type_func)
    plt.ylabel(f'Parameter {type_analyze} value')
    plt.grid()
    plt.show()


def Analyze_for_k_and_d(par_1_or_2, n, input_k_or_d_mas, type_func, type_analyze, iterations=10):
    t_values = [0] * len(input_k_or_d_mas)
    t_val = -1
    samples = []
    for i in range(iterations):
        if type_func == 'stud':
          samples = np.random.standard_t(df=par_1_or_2, size=n)
          samples.sort()
        elif type_func == 'lap':
          samples = np.random.laplace(loc=0, scale=par_1_or_2, size=n)
          samples.sort()
        elif type_func == 'weib':
          samples = np.random.weibull(a=1/2, size=n) * par_1_or_2
          samples.sort()
        elif type_func == 'exp':
          samples = np.random.exponential(scale=1/par_1_or_2, size=n)
          samples.sort()
        counter = 0
        for input_k_or_d in input_k_or_d_mas:
            if type_analyze == 'max_degree':
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
                t_val = max_degree(n, graph)
            elif type_analyze == 'size_max_independent_set':
                graph = create_gd(samples, n, input_k_or_d)
                t_val = size_max_independent_set(n, graph)
            elif type_analyze == 'number_of_connectivity_components':
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
                t_val = number_of_connectivity_components(graph)
            elif type_analyze == 'size_max_clique':
                graph = create_gd(samples, n, input_k_or_d)
                t_val = size_max_clique(graph)
            t_values[counter] += t_val
            counter += 1
    for i in range(len(t_values)):
        t_values[i] /= iterations
    plt.figure()
    plt.plot(input_k_or_d_mas, t_values)
    plt.ylabel(f'Parameter {type_analyze} value')
    if(type_analyze == 'max_degree' or type_analyze == 'number_of_connectivity_components'):
      plt.xlabel(f'Parameter k value,'+ type_func)
    else:
      plt.xlabel(f'Parameter dist value,'+ type_func)
    plt.grid()
    plt.show()

def Analyze_of_n(par_1_or_2, type_func, n_range, input_k_or_d1, type_analyze, iterations=10):
    t_values = []
    t_val = -1
    samples = []
    for n in n_range:
        sum_val = 0
        for i in range(iterations):
            if type_func == 'stud':
                samples = np.random.standard_t(df=par_1_or_2, size=n)
                samples.sort()
            elif type_func == 'lap':
                samples = np.random.laplace(loc=0, scale=par_1_or_2, size=n)
                samples.sort()
            elif type_func == 'weib':
                samples = np.random.weibull(a=1/2, size=n) * par_1_or_2
                samples.sort()
            elif type_func == 'exp':
                samples = np.random.exponential(scale=1/par_1_or_2, size=n)
                samples.sort()
            if type_analyze == 'max_degree':
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d1)
                t_val = max_degree(n, graph)
            elif type_analyze == 'size_max_independent_set':
                graph = create_gd(samples, n, input_k_or_d1)
                t_val = size_max_independent_set(n, graph)
            elif type_analyze == 'number_of_connectivity_components':
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d1)
                t_val = number_of_connectivity_components(graph)
            elif type_analyze == 'size_max_clique':
                graph = create_gd(samples, n, input_k_or_d1)
                t_val = size_max_clique(graph)
            sum_val += t_val
        sum_val /= iterations
        t_values.append(sum_val)
    plt.figure()
    plt.plot(n_range, t_values)
    plt.ylabel(f'Parameter {type_analyze} value')
    plt.xlabel(f'Sample size n value, ' + type_func)
    plt.grid()
    plt.show()


def find_A_1(n, graph_type, input_k_or_d1, iterations):
    values1 = {}
    values2 = {}
    for i in range(iterations):
        samples1 = np.random.standard_t(df=3, size=n)
        samples2 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
        if graph_type == 'knn':
            graph1 = create_gk(samples1.reshape(-1, 1), n, input_k_or_d1)
            graph2 = create_gk(samples2.reshape(-1, 1), n, input_k_or_d1)
            t_val_1 = max_degree(n, graph1)
            t_val_2 = max_degree(n, graph2)
        else:
            graph1 = create_gd(samples1, n, input_k_or_d1)
            graph2 = create_gd(samples2, n, input_k_or_d1)
            t_val_1 = size_max_independent_set(n, graph1)
            t_val_2 = size_max_independent_set(n, graph2)
        values1[t_val_1] = values1.get(t_val_1, 0) + 1
        values2[t_val_2] = values2.get(t_val_2, 0) + 1
    for key in list(values1.keys()):
        values1[key] /= iterations
    for key in list(values2.keys()):
        values2[key] /= iterations
    error = 0
    power = 0
    a = []
    alpha = 1 - 0.05 ** 5
    while values2:
        current_power = 1000
        current_error = 1000
        current_a = None
        for key in values2:
            if values1.get(key, 0) < current_power:
                current_power = values1.get(key, 0)
                current_error = values2.get(key, 0)
                current_a = key
        if current_a is None:
            break
        a.append(current_a)
        power += current_power
        error += current_error
        values1.pop(current_a, None)
        values2.pop(current_a, None)
        if not values2 or error >= alpha:
            break
    print('power A1 = ', power)
    print('error A1 = ', error)
    return a

def find_A_2(n, graph_tipe, input_k_or_d1, iterations):
    values1 = {}
    values2 = {}
    for i in range(iterations):
        samples1 = np.random.weibull(a=1/2, size=n) * 0.31622776601
        samples2 = np.random.exponential(scale=1, size=n)
        if graph_tipe == 'knn':
            graph1 = create_gk(samples1.reshape(-1, 1), n, input_k_or_d1)
            graph2 = create_gk(samples2.reshape(-1, 1), n, input_k_or_d1)
            t_val_1 = number_of_connectivity_components(graph1)
            t_val_2 = number_of_connectivity_components(graph2)
        else:
            graph1 = create_gd(samples1, n, input_k_or_d1)
            graph2 = create_gd(samples2, n, input_k_or_d1)
            t_val_1 = size_max_clique(graph1)
            t_val_2 = size_max_clique(graph2)
        values1[t_val_1] = values1.get(t_val_1, 0) + 1
        values2[t_val_2] = values2.get(t_val_2, 0) + 1
    for key in list(values1.keys()):
        values1[key] /= iterations
    for key in list(values2.keys()):
        values2[key] /= iterations
    error = 0
    power = 0
    a = []
    alpha = 1 - 0.05 ** 5
    while values1:
        current_power = 1000
        current_error = 1000
        current_a = None
        for key in values2:
            if values1.get(key, 0) < current_power:
                current_power = values1.get(key, 0)
                current_error = values2.get(key, 0)
                current_a = key
        if current_a is None:
            break
        a.append(current_a)
        power += current_power
        error += current_error
        values1.pop(current_a, None)
        values2.pop(current_a, None)
        if not values2 or error >= alpha:
            break
    print('power A2 = ', power)
    print('error A2 = ', error)
    return a

#==========================================================================================================
def extract_multiple_features(samples, n, k_or_d, graph_type):
    features = []   
    if graph_type == 'stud':
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
        if dist1 == 'stud':
            samples1 = np.random.standard_t(df=3, size=n)
        elif dist1 == 'lap':
            samples1 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
        elif dist1 == 'weib':
            samples1 = np.random.weibull(a=1/2, size=n) * 0.31622776601
        elif dist1 == 'exp':
            samples1 = np.random.exponential(scale=1, size=n)
            
        features1 = extract_multiple_features(samples1, n, k_or_d, dist1)
        X.append(features1)
        y.append(0)
        
        if dist2 == 'stud':
            samples2 = np.random.standard_t(df=3, size=n)
        elif dist2 == 'lap':
            samples2 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
        elif dist2 == 'weib':
            samples2 = np.random.weibull(a=1/2, size=n) * 0.31622776601
        elif dist2 == 'exp':
            samples2 = np.random.exponential(scale=1, size=n)
            
        features2 = extract_multiple_features(samples2, n, k_or_d, dist1)
        X.append(features2)
        y.append(1) 
    
    if dist1 == 'stud':
      feature_names = ['max_degree', 'size_max_independent_set']
    else:
      feature_names = ['number_of_connectivity_components', 'size_max_clique']
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    if type1 == 'har_analyse':
      y_pred = clf.predict(X_test)
      for name, importance in zip(feature_names, clf.feature_importances_):
        print(f"{name}: {importance:.4f}")
    return clf, df

def analyze_feature_importance_vs_n(n_range, k_or_d, dist1, dist2):
    importance_results = {}
    
    for n in n_range:
        clf, df = build_classifier(n, k_or_d, dist1, dist2,'n_analyse', iterations=50)       
        importance_results[n] = clf.feature_importances_
    
    plt.figure(figsize=(12, 6))
    for feature_idx in range(len(clf.feature_importances_)):
        importances = [importance_results[n][feature_idx] for n in n_range]
        plt.plot(n_range, importances, label=f'Признак {feature_idx}')
    
    plt.xlabel('Размер n')
    plt.ylabel('Важность признака')
    plt.title('Зависимость важности от n')
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
    print('Ошибка первого рода: ', true_false / 2 * n)
    print('Мощность: ', true_true / 2 * n)
    print('Точность: ', (true_true + false_false) / 2 * n)
    return [true_false / 2 * n, true_true / 2 * n, (true_true + false_false) / 2 * n]

def t_classifier_2(classifier, dist, n=50):
    targets = [1] * n + [0] * n
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    for i in range(n):
        samples = np.random.weibull(a=1/2, size=n) * 0.31622776601
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
    print('Ошибка первого рода: ', true_false / 2 * n)
    print('Мощность: ', true_true / 2 * n)
    print('Точность: ', (true_true + false_false) / 2 * n)
    return [true_false / 2 * n, true_true / 2 * n, (true_true + false_false) / 2 * n]
    
def Analyze_of_metric(n_values, k_or_d, dist1, dist2, iterations=50):
    results = []    
    for n in n_values:
        X = []
        y = []
        for i in range(iterations):
            if dist1 == 'stud':
                samples1 = np.random.standard_t(df=3, size=n)
            elif dist1 == 'lap':
                samples1 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
            elif dist1 == 'weib':
                samples1 = np.random.weibull(a=1/2, size=n) * 0.31622776601
            elif dist1 == 'exp':
                samples1 = np.random.exponential(scale=1, size=n)
            
            features1 = extract_multiple_features(samples1, n, k_or_d, dist1)
            X.append(features1)
            y.append(0) 

            if dist2 == 'stud':
                samples2 = np.random.standard_t(df=3, size=n)
            elif dist2 == 'lap':
                samples2 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
            elif dist2 == 'weib':
                samples2 = np.random.weibull(a=1/2, size=n) * 0.31622776601
            elif dist2 == 'exp':
                samples2 = np.random.exponential(scale=1, size=n)
            features2 = extract_multiple_features(samples2, n, k_or_d, dist1)
            X.append(features2)
            y.append(1)  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        classifiers = {
            'Дерево': RandomForestClassifier(n_estimators=100, random_state=42),
            'Логистическая регрессия': LogisticRegression(max_iter=1000, random_state=42),
            'K-ближайших соседей': KNeighborsClassifier(n_neighbors=5)
        }
        
        n_metrics = []
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)            
            n_metrics.append({
                'Классификатор': name,
                'Точность': acc,
                'Precision': report['1']['precision'],
                'Recall': report['1']['recall'],
                'F1-score': report['1']['f1-score']
            })    
        results.append({
            'n': n,
            'Метрики': pd.DataFrame(n_metrics)
        })

    print("\nРезультаты анализа метрик для различных n:")
    for result in results:
      print(f"\nРазмер выборки n = {result['n']}") 
      print(result['Метрики'])
#==========================================================================================================

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
    Analyze_of_parametrs(param_range, n, input_k, 'max_degree', 'stud')
    Analyze_of_parametrs(param_range, n, input_k, 'max_degree', 'lap')

    param_range = np.linspace(2, 30, 100)
    Analyze_of_parametrs(param_range, n, input_d, 'size_max_independent_set', 'stud')
    Analyze_of_parametrs(param_range, n, input_d, 'size_max_independent_set', 'lap')

    param_range = np.linspace(0.5, 10, 70)
    Analyze_of_parametrs(param_range, n, input_k, 'number_of_connectivity_components', 'weib')
    Analyze_of_parametrs(param_range, n, input_k, 'number_of_connectivity_components', 'exp')

    param_range = np.linspace(0.5, 10, 60)
    Analyze_of_parametrs(param_range, n, input_d, 'size_max_clique', 'weib')
    Analyze_of_parametrs(param_range, n, input_d, 'size_max_clique', 'exp')

    # =========== 2 ======================
    print("Анализ четырех функций по k, d и n")
    print("1) max_degree")
    n_range = range(50, 400, 10)
    k_range = range(2, 20)
    Analyze_for_k_and_d(par_1, n, k_range, 'stud', 'max_degree')
    Analyze_for_k_and_d(par_2, n, k_range, 'lap', 'max_degree')
    Analyze_of_n(par_1, 'stud', n_range, input_k, 'max_degree')
    Analyze_of_n(par_2, 'lap', n_range, input_k, 'max_degree')

    print("2) size_max_independent_set")
    d_range = np.linspace(0.05, 10, 30)
    Analyze_for_k_and_d(par_2, n, d_range, 'lap', 'size_max_independent_set')
    Analyze_for_k_and_d(par_1, n, d_range, 'stud', 'size_max_independent_set')
    Analyze_of_n(par_2, 'lap', n_range, input_d, 'size_max_independent_set')
    Analyze_of_n(par_2, 'stud', n_range, input_d, 'size_max_independent_set')

    print("3) number_of_connectivity_components")
    n_range = range(50, 100, 2)
    k_range = range(2, 20)
    Analyze_for_k_and_d(par_3, n, k_range, 'exp', 'number_of_connectivity_components')
    Analyze_for_k_and_d(par_4, n, k_range, 'weib', 'number_of_connectivity_components')
    Analyze_of_n(par_3, 'exp', n_range, input_k, 'number_of_connectivity_components')
    Analyze_of_n(par_4, 'weib', n_range, input_k, 'number_of_connectivity_components')

    print("4) size_max_clique")
    d_range = np.linspace(0.05, 10, 30)
    Analyze_for_k_and_d(par_3, n, d_range, 'exp', 'size_max_clique')
    Analyze_for_k_and_d(par_4, n, d_range, 'weib', 'size_max_clique')
    Analyze_of_n(par_3, 'exp', n_range, input_d, 'size_max_clique')
    Analyze_of_n(par_4, 'weib', n_range, input_d, 'size_max_clique')

    # =========== 3 ======================
    print("Построение множества А1 и А2")

    n = 300
    k = 5
    d = 0.2
    iterations = 1000

    A1_knn = find_A_1(n, 'knn', k, iterations)
    A1_dist = find_A_1(n, 'dist', d, iterations)
    A2_knn = find_A_2(n, 'knn', k, iterations)
    A2_dist = find_A_2(n, 'dist', d, iterations)

    # ============= Часть 2 =================

    # =========== 1 ======================
    print("\n")
    print("Исследование важности характеристик")
    print("\n")  
    print("1)Важность признаков у stud и lap")
    clf_knn, df_knn = build_classifier(n, input_d, 'stud', 'lap', 'har_analyse')
    print("\n")
    print("2)Важность признаков у exp и weib")   
    clf_dist, df_dist = build_classifier(n, input_d, 'weib', 'exp', 'har_analyse')
    print("\n")
    print("Анализ важности признаков в зависимости от размера выборки:")
    n_range = range(20, 201, 40)
    analyze_feature_importance_vs_n(n_range, input_d, 'stud', 'lap')
    analyze_feature_importance_vs_n(n_range, input_d, 'weib', 'exp')

    # =========== 2 ======================
    n_range = [25, 100, 500]
    Analyze_of_metric(n_range, input_d, 'stud', 'lap')
    Analyze_of_metric(n_range, input_d, 'weib', 'exp')
    # =========== 3 ======================
