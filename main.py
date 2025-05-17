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

def t_d(n, gd):
    return 0

def t_k(n, gk):
    degrees = np.sum(gk, axis=1)
    return np.max(degrees) if n > 0 else 0

def td_2(adj_matrix):
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

def t_k_2(data):
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


def first_analyze(param_range, n, input_k_or_d, type_analyze):
    t_values = []
    for param in param_range:
        if type_analyze == 'knn':
            samples = np.random.standard_t(df=param, size=n)
            graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
            t_val = t_k(n, graph)
        elif type_analyze == 'd':
            samples = np.random.laplace(loc=0, scale=param, size=n)
            graph = create_gd(samples, n, input_k_or_d)
            t_val = t_d(n, graph)
        elif type_analyze == 'knn_2':
            samples = np.random.weibull(a=1/2, size=n) * param
            graph = create_gd(samples, n, input_k_or_d)
            t_val = t_k_2(graph)
        else:
            samples = np.random.exponential(scale=1/param, size=n)
            graph = create_gd(samples, n, input_k_or_d)
            t_val = td_2(graph)
        t_values.append(t_val)

    plt.figure()
    plt.plot(param_range, t_values)
    plt.xlabel('Parameter value')
    plt.ylabel(f'T value')
    plt.grid()
    plt.show()

def second_analyze(par_1_or_2, n, input_k_or_d_mas, type_analyze):
    t_values = []
    if type_analyze == 'knn':
      samples = np.random.standard_t(df=par_1_or_2, size=n)
    elif type_analyze == 'd':
      samples = np.random.laplace(loc=0, scale=par_1_or_2, size=n)
    elif type_analyze == 'knn_2':
      samples = np.random.weibull(a=1/2, size=n) * par_1_or_2
    else:
      samples = np.random.exponential(scale=1/par_1_or_2, size=n)
    for input_k_or_d in input_k_or_d_mas:
        if type_analyze == 'knn':
            graph = create_gk(samples.reshape(-1, 1), n, input_k_or_d)
            t_val = t_k(n, graph)
        elif type_analyze == 'd':
            graph = create_gd(samples, n, input_k_or_d)
            t_val = t_d(n, graph)
        elif type_analyze == 'knn_2':
            graph = create_gd(samples, n, input_k_or_d)
            t_val = t_k_2(graph)
        else:
            graph = create_gd(samples, n, input_k_or_d)
            t_val = td_2(graph)
        t_values.append(t_val)

    plt.figure()
    plt.plot(input_k_or_d_mas, t_values)
    plt.xlabel('Parameter value')
    plt.ylabel(f'T value')
    plt.grid()
    plt.show()


if __name__ == "__main__":
  '''
  par_1 = float(input("Введите первый параметр распределения: "))
  par_2 = float(input("Введите второй параметр распределения: "))
  par_3 = float(input("Введите третий параметр распределения: "))
  par_4 = float(input("Введите четвертый параметр распределения: "))
  '''
  par_1 = 3
  par_2 = 0.70710678118
  par_3 = 0.31622776601
  par_4 = 1
  E_mas = []
  input_n = int(input("Введите n: "))

  input_k = int(input("Введите k: "))
  input_d = float(input("Введите d: "))
# =========== 1 ====================

# =========== 2 ======================
  n = 100
  print("First_part")
  param_range = np.linspace(2, 30, 100)
  first_analyze(param_range, n, input_k, 'knn')

  param_range = np.linspace(0.5, 10, 70)
  first_analyze(param_range, n, input_d, 'd')

  param_range = np.linspace(2, 30, 50)
  first_analyze(param_range, n, input_k, 'knn_2')

  param_range = np.linspace(0.5, 10, 70)
  first_analyze(param_range, n, input_d, 'd_2')

# =========== 3 ======================
  print("Second_part")
  k_range = range(2, 20)
  second_analyze(par_1, n, k_range, 'knn')

  d_range = np.linspace(0.05, 10, 30)
  second_analyze(par_2, n, d_range, 'd')

  k_range = range(2, 20)
  second_analyze(par_1, n, k_range, 'knn_2')

  d_range = np.linspace(0.05, 10, 30)
  second_analyze(par_2, n, d_range, 'd_2')
  # =========== 4 ======================
  print("3_part")
