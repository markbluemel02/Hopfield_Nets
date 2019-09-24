import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from learning_rules import *

def random_state(p, n):
    return np.array([np.sign(np.random.rand()-p) for i in range(n)])

def random_state_(p, n):
    return np.array([np.random.choice([-0.9999, 0.9999], 1, [1 - p, p])[0] for i in range(n)])

def introduce_random_flips(pattern, k):
    new_pattern = deepcopy(pattern)
    inds = np.random.choice(np.arange(len(new_pattern)), k, replace=False)
    for i in range(k):
        new_pattern[inds[i]] *= -1
    return new_pattern

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class Hopfield_network():
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        Y = np.random.randn(self.num_neurons, self.num_neurons)
        R = (Y + Y.T) / 2
        R[np.arange(num_neurons), np.arange(num_neurons)] = np.zeros(num_neurons)
        self.weights = (1 / self.num_neurons) * R
        # self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.biases = np.zeros(self.num_neurons)
        self.hidden_state = np.random.rand(num_neurons)-0.5
        self.state = np.sign(self.hidden_state)

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = (self.weights @ self.state.reshape(-1, 1) + self.biases.reshape(-1, 1)).flatten()
            self.state = np.sign(self.hidden_state)
        elif sync == False:
            for i in range(self.num_neurons):
                self.hidden_state[i] = np.dot(self.weights[i, :], self.state) + self.biases[i]
                self.state[i] = np.sign(self.hidden_state[i])
        else:
            raise AttributeError('sync variable can take only boolean values')
        return None

    def learn_patterns(self, patterns, rule, options):
        patterns = deepcopy(np.array(patterns).reshape(-1, len(patterns[0])))
        if rule == 'Hebb':
            self.weights, self.biases = hebbian_lr(self.num_neurons, patterns, self.weights, self.biases,  **options)
        elif rule == 'pseudoinverse':
            self.weights, self.biases = pseudoinverse(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule =='Storkey2ndOrder':
            self.weights, self.biases = storkey_2_order(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'StorkeySimplified':
            self.weights, self.biases = storkey_simplified(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'StorkeyAsymm':
            self.weights, self.biases = storkey_asymmetric(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule =='StorkeyOriginal':
            self.weights, self.biases = storkey_original(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'OptimisedQP':
            self.weights = optimisation_quadratic(self.num_neurons, patterns, self.weights, self.biases, options)
        elif rule == 'OptimisedSequentialQP':
            self.weights, self.biases = optimisation_sequential_quadratic(self.num_neurons, patterns, self.weights, self.biases, options)
        elif rule == 'OptimisedLP':
            self.weights, self.biases = optimisation_linear(self.num_neurons, patterns, self.weights, self.biases, options)
        elif rule == 'OptimisedQPincremental':
            self.weights, self.biases = optimisation_incremental_quadratic(self.num_neurons, patterns, self.weights, self.biases, options)
        elif rule == 'OptimisedLPincremental':
            self.weights, self.biases = optimisation_incremental_linear(self.num_neurons, patterns, self.weights, self.biases, options)
        elif rule == 'DescentL2':
            self.weights, self.biases = descent_l2_norm(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'DescentL2Solver':
            self.weights, self.biases = descent_l2_with_solver(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'DescentL2Symm':
            self.weights, self.biases = descent_l2_symm(self.num_neurons, patterns, self.weights, self.biases, options)
        elif rule == 'DescentCrossentropy':
            self.weights, self.biases = descent_crossentropy(self.num_neurons, patterns, self.weights, self.biases, options)
        elif rule == 'DescentHamming':
            self.weights, self.biases = descent_Hamming(self.num_neurons, patterns, self.weights, self.biases, options)
        elif rule == 'DescentOverlap':
            self.weights, self.biases = descent_overlap(self.num_neurons, patterns, self.weights, self.biases, **options)
        else:
            raise ValueError('the specified learning rule is not implemented')
        return None

    def set_params(self, new_weights, new_biases):
        self.weights = new_weights
        self.biases = new_biases
        return None

    def retrieve_pattern(self, initial_state, sync, time, record = False):
        global data
        self.state = deepcopy(initial_state)
        t = 0
        if record == True:
            data = dict()
            data['hidden_variables'] = []
            data['state'] = []
        while t < time:
            self.update_state(sync)
            t += 1
            if record == True:
                data['hidden_variables'].append(deepcopy(self.hidden_state))
                data['state'].append(deepcopy(self.state))
        if record == True:
            data['hidden_variables'] = np.array(data['hidden_variables']).T
            data['state'] = np.array(data['state']).T
            return self.state, data
        else:
            return self.state

class Boltzmann_machine(Hopfield_network):
    def __init__(self, num_neurons, T_max, T_decay):
        super().__init__(num_neurons)
        self.T = T_max
        self.T_max = T_max
        self.T_decay = T_decay

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = np.matmul(self.weights,self.state) + self.biases
            h_tmp = sigmoid(self.hidden_state/self.T)
            self.state = np.array([(1 if np.random.rand() < h_tmp[i] else -1) for i in range(len(h_tmp))])
        elif sync == False:
            for i in range(num_neurons):
                self.hidden_state[i] = np.dot(self.weights[i, :], self.state)
                self.state[i] = 1 if np.random.rand() < sigmoid(self.hidden_state[i] / self.T) else -1
        else:
            raise AttributeError('sync variable can take only boolean values')
        return None

    def retrieve_pattern(self, new_state, sync, time = None, record = False):
        self.T = self.T_max
        self.state = new_state
        if record == True:
            data = dict()
            data['hidden_variables'] = []
            data['state'] = []
        while self.T > 0.0001:
            self.update_state(sync)
            self.T *= self.T_decay
            if record == True:
                data['hidden_variables'].append(deepcopy(self.hidden_state))
                data['state'].append(deepcopy(self.state))
        if record == True:
            data['hidden_variables'] = np.array(data['hidden_variables']).T
            data['state'] = np.array(data['state']).T
            return self.state, data
        else:
            return self.state

class Continuous_HN_DS(Hopfield_network):
    def __init__(self, num_neurons, dt):
        super().__init__(num_neurons)
        self.dt = dt
        self.fr = (self.state + 1)/2

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state += self.dt * (-self.hidden_state + np.matmul(self.weights, self.state) + self.biases)
            self.state = np.sign(self.hidden_state)
        else:
            for i in range(self.num_neurons):
                self.hidden_state[i] += self.dt * (-self.hidden_state[i] + np.dot(self.weights[i, :], self.state) + self.biases[i])
                self.state[i] = np.sign(self.hidden_state[i])
        self.fr = (self.state + 1) / 2
        return None

class Continuous_HN(Hopfield_network):
    def __init__(self, num_neurons, slope, dt):
        super().__init__(num_neurons)
        self.slope = slope
        self.dt = dt
        self.fr = (self.state + 1)/2

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state += self.dt * (-self.hidden_state + np.matmul(self.weights, self.state) + self.biases)
            self.state = 2 * sigmoid(self.hidden_state * self.slope) - 1
        else:
            for i in range(self.num_neurons):
                self.hidden_state[i] += self.dt * (-self.hidden_state[i] + np.dot(self.weights[i, :], self.state) + self.biases[i])
                self.state[i] = 2 * sigmoid(self.hidden_state[i] * self.slope) - 1
        self.fr = (self.state + 1) / 2
        return None

class Continuous_HN_DS_ad(Continuous_HN_DS):
    def __init__(self, num_neurons, dt, a, b):
        super().__init__(num_neurons, dt)
        self.u = np.zeros(self.num_neurons)
        self.a = a
        self.b = b

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state += self.dt * (-self.hidden_state + np.matmul(self.weights, self.state) - self.u + self.biases)
            self.u += self.dt * self.a * (self.b * self.hidden_state - self.u)
            self.state = np.sign(self.hidden_state)
        else:
            for i in range(self.num_neurons):
                self.hidden_state[i] += self.dt * (-self.hidden_state[i] + np.dot(self.weights[i, :], self.state) - self.u[i] + self.biases[i])
                self.u[i] += self.dt * self.a * (self.b * self.hidden_state[i] - self.u[i])
                self.state[i] = np.sign(self.hidden_state[i])
        self.fr = (self.state + 1) / 2
        return None

class Continuous_HN_ad(Continuous_HN):
    def __init__(self, num_neurons, slope, dt, a, b):
        super().__init__(num_neurons, slope, dt)
        self.u = np.zeros(self.num_neurons)
        self.a = a
        self.b = b

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state += self.dt * (-self.hidden_state + np.matmul(self.weights, self.state) - self.u + self.biases)
            self.u += self.dt * self.a * (self.b * self.hidden_state - self.u)
            self.state = sigmoid(self.hidden_state * self.slope)
        else:
            for i in range(self.num_neurons):
                self.hidden_state[i] += self.dt * (-self.hidden_state[i] + np.dot(self.weights[i, :], self.state) - self.u[i] + self.biases[i])
                self.u[i] += self.dt * self.a * (self.b * self.hidden_state[i] - self.u[i])
                self.state[i] = sigmoid(self.hidden_state[i] * self.slope)
        self.fr = (self.state + 1) / 2
        return None

if __name__ == '__main__':
    num_neurons = 100
    num_patterns = 10
    sync = True
    flips = 20
    time = 100
    rule = 'DescentL2Solver'
    options = {'rule': rule, 'sc': True, 'symm': True, 'incremental': True, 'tol': 1e-3, 'lambda': 1.0, 'activation_function' : 'identity'}
    HN = Hopfield_network(num_neurons=num_neurons)
    patterns = [random_state(0.5, num_neurons) for i in range(num_patterns)]
    HN.learn_patterns(patterns, options)
    pattern_r = introduce_random_flips(patterns[3], flips)

    print('\nSimilarity with different patterns (before): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(pattern_r - patterns[i], 2))

    retrieved_pattern, data = HN.retrieve_pattern(pattern_r, sync, time, record = True)
    print('\nSimilarity with different patterns (after): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(retrieved_pattern - patterns[i], 2))

    fig1 = plt.figure()
    plt.plot(data['hidden_variables'].T)
    plt.show()
    fig3 = plt.figure()
    similarities = np.array([[np.dot(data['state'][:,j],patterns[i])/num_neurons for i in range(len(patterns))] for j in range(time)])
    plt.plot(similarities)
    plt.show()
    # print('\nSimilarity of signs with different patterns (after): ')
    # for i in range(len(patterns)):
    #     print(np.linalg.norm(np.sign(retrieved_pattern) - patterns[i], 2))

