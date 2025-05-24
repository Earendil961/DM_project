import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

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
    for i in range(n):
        for j in range(1, k):
            neighbor = indices[i][j]
            gk[i][neighbor] = True
            gk[neighbor][i] = True
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
    if n == 0:
        return 0
    reach = [row.copy() for row in data]
    for i in range(n):
        reach[i][i] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    visited = [False] * n
    count = 0
    for i in range(n):
        if not visited[i]:
            count += 1
            for j in range(n):
                if reach[i][j] and reach[j][i]:
                    visited[j] = True
    return count


def Analyze_of_parametrs(param_range, n, input_k_or_d, type_analyze):
    t_values = []
    for param in param_range:
        if type_analyze == 'stud':
            samples = np.random.standard_t(df=param, size=n)
            graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
            t_val = max_degree(n, graph)
        elif type_analyze == 'lap':
            samples = np.random.laplace(loc=0, scale=param, size=n)
            graph = create_gd(samples, n, input_k_or_d)
            t_val = size_max_independent_set(n, graph)
        elif type_analyze == 'weib':
            samples = np.random.weibull(a=1/2, size=n) * param
            graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
            t_val = number_of_connectivity_components(graph)
        else:
            samples = np.random.exponential(scale=1/param, size=n)
            graph = create_gd(samples, n, input_k_or_d)
            t_val = size_max_clique(graph)
        t_values.append(t_val)

    plt.figure()
    plt.plot(param_range, t_values)
    plt.xlabel('Parameter k or d value')
    plt.ylabel(f'T value')
    plt.grid()
    plt.show()


def Analyze_for_k_and_d(par_1_or_2, n, input_k_or_d_mas, type_analyze):
    t_values = []
    if type_analyze == 'stud':
      samples = np.random.standard_t(df=par_1_or_2, size=n)
    elif type_analyze == 'lap':
      samples = np.random.laplace(loc=0, scale=par_1_or_2, size=n)
    elif type_analyze == 'weib':
      samples = np.random.weibull(a=1/2, size=n) * par_1_or_2
    else:
      samples = np.random.exponential(scale=1/par_1_or_2, size=n)
    for input_k_or_d in input_k_or_d_mas:
        if type_analyze == 'stud':
            graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
            t_val = max_degree(n, graph)
        elif type_analyze == 'lap':
            graph = create_gd(samples, n, input_k_or_d)
            t_val = size_max_independent_set(n, graph)
        elif type_analyze == 'weib':
            graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
            t_val = number_of_connectivity_components(graph)
        else:
            graph = create_gd(samples, n, input_k_or_d)
            t_val = size_max_clique(graph)
        t_values.append(t_val)

    plt.figure()
    plt.plot(input_k_or_d_mas, t_values)
    plt.xlabel('Parameter value')
    plt.ylabel(f'T value')
    plt.grid()
    plt.show()

def Analyze_of_n(par_1_or_2, n, type_analyze, n_range, input_k_or_d1):
    t_values1 = []
    for n1 in n_range:
        if type_analyze == 'stud':
            samples1 = np.random.standard_t(df=par_1_or_2, size=n1)
            graph1 = create_gk(samples1.reshape(-1, 1), n1, input_k_or_d1)
            t_val = max_degree(n1, graph1)
        elif type_analyze == 'lap':
            samples1 = np.random.laplace(loc=0, scale=par_1_or_2, size=n1)
            graph1 = create_gd(samples1, n1, input_k_or_d1)
            t_val = size_max_independent_set(n1, graph1)
        elif type_analyze == 'weib':
            samples1 = np.random.weibull(a=1/2, size=n1) * par_1_or_2
            graph1 = create_gk(samples1.reshape(-1, 1), n1, input_k_or_d1)
            t_val = number_of_connectivity_components(graph1)
        else:
            samples1 = np.random.exponential(scale=1/par_1_or_2, size=n1)
            graph1 = create_gd(samples1, n1, input_k_or_d1)
            t_val = size_max_clique(graph1)
        t_values1.append(t_val)

    plt.figure()
    plt.plot(n_range, t_values1)
    plt.xlabel('N value')
    plt.ylabel(f'T value')
    plt.grid()
    plt.show()


if __name__ == "__main__":

  par_1 = 3
  par_2 = 0.70710678118
  par_3 = 0.31622776601
  par_4 = 1

  input_k = int(input("Введите k: "))
  input_d = float(input("Введите d: "))

  # =========== 1 ======================
  n = 100
  print("Анализ четырех функций по их параметрам")
  param_range = np.linspace(2, 30, 100)
  Analyze_of_parametrs(param_range, n, input_k, 'stud')

  param_range = np.linspace(0.5, 10, 70)
  Analyze_of_parametrs(param_range, n, input_d, 'lap')

  param_range = np.linspace(2, 30, 45)
  Analyze_of_parametrs(param_range, n, input_k, 'weib')

  param_range = np.linspace(0.5, 10, 60)
  Analyze_of_parametrs(param_range, n, input_d, 'exp')

  # =========== 2 ======================
  print("Анализ четырех функций по k, d и n")
  print("1) stud")
  n_range = range(50, 500, 50)
  k_range = range(2, 20)
  Analyze_for_k_and_d(par_1, n, k_range, 'stud')
  Analyze_of_n(par_1, n, 'stud', n_range, input_k)

  print("2) lap")
  d_range = np.linspace(0.05, 10, 30)
  Analyze_for_k_and_d(par_2, n, d_range, 'lap')
  Analyze_of_n(par_2, n, 'lap', n_range, input_d)
 
  print("3) weib")
  n_range = range(50, 100, 5)
  k_range = range(2, 20)
  Analyze_for_k_and_d(par_3, n, k_range, 'weib')
  Analyze_of_n(par_3, n, 'weib', n_range, input_k)

  print("4) exp")
  d_range = np.linspace(0.05, 10, 30)
  Analyze_for_k_and_d(par_4, n, d_range, 'exp')
  Analyze_of_n(par_4, n, 'exp', n_range, input_d)
  # =========== 3 ======================
  print("Построение множества А")

  # ============= Часть 2 =================
