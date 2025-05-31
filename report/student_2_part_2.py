from main import *

def Analyze_of_parametrs_M(param_range, n, input_k_or_d, type_analyze, type_func, iterations=10):
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
        t_values.append((sum_val / iterations)**(5/2))
    plt.figure()
    plt.plot(param_range, t_values)
    plt.xlabel('Parameter value, ' + type_func)
    plt.ylabel(f'Parameter {type_analyze} value')
    plt.grid()
    plt.show()

param_range = np.linspace(0.5, 10, 70)
n = 100
input_d = 0.2
Analyze_of_parametrs_M(param_range, n, input_d, 'size_max_clique', 'exp')
