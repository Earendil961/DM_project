import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


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
            if gd[i][j] == 1:
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
            if gk[i][j] == 1:
                arr[i] = arr[i] + 1
    m = 0
    for i in range(n):
        if m < arr[i]:
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


def analyze_of_params(
    param_range, n, input_k_or_d, type_analyze, type_func, iterations=10
):
    t_values = []
    t_val = -1
    samples = []
    for param in param_range:
        sum_val = 0
        for i in range(iterations):
            if type_func == "stud":
                samples = np.random.standard_t(df=param, size=n)
                samples.sort()
            elif type_func == "lap":
                samples = np.random.laplace(loc=0, scale=param, size=n)
                samples.sort()
            elif type_func == "weib":
                samples = np.random.weibull(a=1 / 2, size=n) * param
                samples.sort()
            elif type_func == "exp":
                samples = np.random.exponential(scale=1 / param, size=n)
                samples.sort()
            if type_analyze == "max_degree":
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
                t_val = max_degree(n, graph)
            elif type_analyze == "size_max_independent_set":
                graph = create_gd(samples, n, input_k_or_d)
                t_val = size_max_independent_set(n, graph)
            elif type_analyze == "number_of_connectivity_components":
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
                t_val = number_of_connectivity_components(graph)
            elif type_analyze == "size_max_clique":
                graph = create_gd(samples, n, input_k_or_d)
                t_val = size_max_clique(graph)
            sum_val += t_val
        t_values.append(sum_val / iterations)
    plt.figure()
    plt.plot(param_range, t_values)
    plt.xlabel("Parameter value, " + type_func)
    plt.ylabel(f"Parameter {type_analyze} value")
    plt.grid()
    plt.show()


def analyze_for_k_and_d(
    par_1_or_2, n, input_k_or_d_mas, type_func, type_analyze, iterations=10
):
    t_values = [0] * len(input_k_or_d_mas)
    t_val = -1
    samples = []
    for i in range(iterations):
        if type_func == "stud":
            samples = np.random.standard_t(df=par_1_or_2, size=n)
            samples.sort()
        elif type_func == "lap":
            samples = np.random.laplace(loc=0, scale=par_1_or_2, size=n)
            samples.sort()
        elif type_func == "weib":
            samples = np.random.weibull(a=1 / 2, size=n) * par_1_or_2
            samples.sort()
        elif type_func == "exp":
            samples = np.random.exponential(scale=1 / par_1_or_2, size=n)
            samples.sort()
        counter = 0
        for input_k_or_d in input_k_or_d_mas:
            if type_analyze == "max_degree":
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
                t_val = max_degree(n, graph)
            elif type_analyze == "size_max_independent_set":
                graph = create_gd(samples, n, input_k_or_d)
                t_val = size_max_independent_set(n, graph)
            elif type_analyze == "number_of_connectivity_components":
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
                t_val = number_of_connectivity_components(graph)
            elif type_analyze == "size_max_clique":
                graph = create_gd(samples, n, input_k_or_d)
                t_val = size_max_clique(graph)
            t_values[counter] += t_val
            counter += 1
    for i in range(len(t_values)):
        t_values[i] /= iterations
    plt.figure()
    plt.plot(input_k_or_d_mas, t_values)
    plt.ylabel(f"Parameter {type_analyze} value")
    if (
        type_analyze == "max_degree"
        or type_analyze == "number_of_connectivity_components"
    ):
        plt.xlabel(f"Parameter k value," + type_func)
    else:
        plt.xlabel(f"Parameter dist value," + type_func)
    plt.grid()
    plt.show()


def analyze_of_n(
    par_1_or_2, type_func, n_range, input_k_or_d1, type_analyze, iterations=10
):
    t_values = []
    t_val = -1
    samples = []
    for n in n_range:
        sum_val = 0
        for i in range(iterations):
            if type_func == "stud":
                samples = np.random.standard_t(df=par_1_or_2, size=n)
                samples.sort()
            elif type_func == "lap":
                samples = np.random.laplace(loc=0, scale=par_1_or_2, size=n)
                samples.sort()
            elif type_func == "weib":
                samples = np.random.weibull(a=1 / 2, size=n) * par_1_or_2
                samples.sort()
            elif type_func == "exp":
                samples = np.random.exponential(scale=1 / par_1_or_2, size=n)
                samples.sort()
            if type_analyze == "max_degree":
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d1)
                t_val = max_degree(n, graph)
            elif type_analyze == "size_max_independent_set":
                graph = create_gd(samples, n, input_k_or_d1)
                t_val = size_max_independent_set(n, graph)
            elif type_analyze == "number_of_connectivity_components":
                graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d1)
                t_val = number_of_connectivity_components(graph)
            elif type_analyze == "size_max_clique":
                graph = create_gd(samples, n, input_k_or_d1)
                t_val = size_max_clique(graph)
            sum_val += t_val
        sum_val /= iterations
        t_values.append(sum_val)
    plt.figure()
    plt.plot(n_range, t_values)
    plt.ylabel(f"Parameter {type_analyze} value")
    plt.xlabel(f"Sample size n value, " + type_func)
    plt.grid()
    plt.show()


def find_A_1(n, graph_type, input_k_or_d1, iterations):
    values1 = {}
    values2 = {}
    for i in range(iterations):
        samples1 = np.random.standard_t(df=3, size=n)
        samples2 = np.random.laplace(loc=0, scale=0.70710678118, size=n)
        if graph_type == "knn":
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
    alpha = 1 - 0.05**5
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
    print("power A1 = ", power)
    print("error A1 = ", error)
    return a


def find_A_2(n, graph_tipe, input_k_or_d1, iterations):
    values1 = {}
    values2 = {}
    for i in range(iterations):
        samples1 = np.random.weibull(a=1 / 2, size=n) * 0.31622776601
        samples2 = np.random.exponential(scale=1, size=n)
        if graph_tipe == "knn":
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
    alpha = 1 - 0.05**5
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
    print("power A2 = ", power)
    print("error A2 = ", error)
    return a
