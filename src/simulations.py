import numpy as np
import pickle
from Hopfield_Network import *
from utils import *
from matplotlib import pyplot as plt
from copy import deepcopy
from visualisation import flips_and_patterns_contour_plot
import sys
import warnings
import argparse

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def flips_and_patterns(num_neurons, num_of_flips, num_of_patterns, num_repetitions, params, plot = False):
    '''
    :param num_neurons:
    :param num_of_flips:
    :param num_of_patterns:
    :param num_repetitions:
    :param params:
    :param plot:
    :return:
    '''
    rule = params['rule']
    learning_options = params['learning_options']
    retrieval_options = params['retrieval_options']
    time = retrieval_options['time_of_retrieval']
    sync = retrieval_options['sync']
    file_name = f'../data/flips_and_patterns_{get_postfix(rule, learning_options, num_neurons, num_of_patterns, num_repetitions)}.pkl'
    results = np.zeros((num_of_patterns, num_of_flips, num_repetitions))
    for r in range(num_repetitions):
        HN = Hopfield_network(num_neurons=num_neurons)
        print(f'repetition number {r}')
        for n_p in range(1,num_of_patterns+1):
            print(f'learning {n_p} patterns')
            patterns = deepcopy([random_state(0.5, num_neurons, values=[-1, 1]) for i in range(n_p)])

            R = np.random.randn(num_neurons, num_neurons)
            R = (R + R.T) / 2
            R[np.arange(num_neurons), np.arange(num_neurons)] = np.zeros(num_neurons)

            HN.set_params((1 / num_neurons) * R, np.zeros(num_neurons))

            HN.learn_patterns(patterns, rule, learning_options)
            for n_f in range(1, num_of_flips+1):
                num = np.random.randint(len(patterns))
                true_pattern = deepcopy(patterns[num])
                pattern_r = deepcopy(introduce_random_flips(true_pattern, n_f, values = [-1, 1]))
                retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
                overlap = (1 / num_neurons) * np.dot(retrieved_pattern, true_pattern)
                results[n_p - 1, n_f - 1, r] = overlap
    pickle.dump(results, open(file_name,'wb+'))
    if plot == True:
        file_name = f'../data/flips_and_patterns_{get_postfix(rule, learning_options, num_neurons, num_of_patterns, num_repetitions)}.pkl'
        flips_and_patterns_contour_plot(file_name)
    return None

def weights_distribution_plot(num_neurons, num_of_patterns, params):
    rule = params['rule']
    options = params['learning_options']
    HN = Hopfield_network(num_neurons=num_neurons)
    patterns = deepcopy([random_state(0.5, num_neurons) for i in range(num_of_patterns)])
    HN.learn_patterns(patterns, rule, options)

    w = HN.weights.flatten()
    fig = plt.figure()
    _ = plt.hist(w, density=True, facecolor='g', alpha=0.75, bins='auto')  # arguments are passed to np.histogram
    plt.xlabel('Weight Value')
    plt.ylabel('Empirical Probability Density')
    plt.title(f'Weight distribution (N = {num_neurons}, Num patterns = {num_of_patterns}, rule = {rule})')
    plt.grid(True)
    # plt.xticks(np.arange(np.round(min(w),-1), np.round(max(w) + 1,-1), 5))
    plt.savefig(f'../imgs/{rule}/WeightDistr_(N = {num_neurons}_Num patterns = {num_of_patterns}_rule = {rule})')
    # plt.show()
    plt.close(fig=fig)
    return None

def custom_flips_and_patterns(num_neurons, num_of_flips,n_pattern_list, num_repetitions, params,seed=None,plot = False):
    '''
    :param num_neurons:
    :param num_of_flips:
    :param n_pattern_list: an iterator over the patterns used
    :param num_repetitions:
    :param params:
    :param plot:
    :return:
    '''
    rule = params['rule']
    learning_options = params['learning_options']
    retrieval_options = params['retrieval_options']
    time = retrieval_options['time_of_retrieval']
    sync = retrieval_options['sync']
    maximum_patterns = np.max(n_pattern_list)
    # parent = '/user/mblueme/u26551/.project/dir.project/mark/data/Tolmachev'
    parent = '/home/nst/mbluemel/Repos/Hopfield_Nets'
    seed_string = ''
    if seed is not None:
        seed_string = f'seed_{seed}'
    file_name = f'{parent}/data/test/flips_and_patterns_{get_postfix(rule, learning_options, num_neurons, maximum_patterns, num_repetitions)}_{seed_string}.pkl'
    
    results = np.zeros((len(n_pattern_list), num_of_flips, num_repetitions))
    for r in range(num_repetitions):
        HN = Hopfield_network(num_neurons=num_neurons)
        print(f'repetition number {r}')
        for i,n_p in enumerate(n_pattern_list):
            print(f'learning {n_p} patterns')
            patterns = deepcopy([random_state(0.5, num_neurons, values=[-1, 1]) for i in range(n_p)])

            R = np.random.randn(num_neurons, num_neurons)
            R = (R + R.T) / 2
            R[np.arange(num_neurons), np.arange(num_neurons)] = np.zeros(num_neurons)

            HN.set_params((1 / num_neurons) * R, np.zeros(num_neurons))

            HN.learn_patterns(patterns, rule, learning_options)
            for n_f in range(1, num_of_flips+1):
                num = np.random.randint(len(patterns))
                true_pattern = deepcopy(patterns[num])
                pattern_r = deepcopy(introduce_random_flips(true_pattern, n_f, values = [-1, 1]))
                retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
                overlap = (1 / num_neurons) * np.dot(retrieved_pattern, true_pattern)
                results[i , n_f - 1, r] = overlap
    pickle.dump(results, open(file_name,'wb+'))
    if plot == True:
        file_name = f'{parent}/data/flips_and_patterns_{get_postfix(rule, learning_options, num_neurons, num_of_patterns, num_repetitions)}.pkl'
        flips_and_patterns_contour_plot(file_name)
    return None

def init_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--num_points", type=int, default=21)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    seed = args.seed # int or None
    if seed is not None:
        init_seed(seed)
        print(f'Using custom seed {seed}.')
    # run simulations
    num_neurons = args.neurons
    num_of_flips = num_neurons//2
    # num_of_patterns = 100
    num_repetitions = args.reps
    num_points = args.num_points


    rules = [
            #non-incremental
            'Hebb',
            # 'Hebb',
            #'Storkey',
            #'Pseudoinverse',
            #'KrauthMezard',
            #'DescentExpBarrier',
            #'DescentExpBarrierSI',
            #'DescentL1',
            # 'DescentL2',
            #  'DescentL2',
            # 'GardnerKrauthMezard'
            #incremental
            # 'Hebb',
            # 'Storkey',
            # 'DiederichOpperI',
            # 'DiederichOpperII',
            #'DescentExpBarrier',
            # 'DescentExpBarrierSI',
            # 'DescentL1',
            #'DescentL2',
            # 'GardnerKrauthMezard',

            # for sc effects
            # 'Hebb',
            # 'Storkey',
            # 'DescentL2',
            # 'DescentL1',
            # 'GardnerKrauthMezard',
            # 'DescentExpBarrierSI'

            # Infomorphic rule
            # 'Infomorphic',
            # 'Infomorphic', #optimized
            # 'MPF'
    ]
    options = [# Non-incremental
                #{'incremental' : False, 'sc' : True },  #Hebbian
                {'incremental' : False, 'sc' : False },  #Hebbian
                #{'incremental' : False, 'sc': True},  # Storkey
                #{},  #Pseudoinverse
                #{'sc' : True, 'lr': 1e-2, 'maxiter': 200},  # Krauth-Mezard
                #{'sc' : True, 'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentExpBarrier
                #{'sc' : True, 'incremental' : False, 'tol' : 1e-3, 'lmbd': 0.5}, # DescentExpBarrierSI
                #{'sc' : True, 'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL1
                # {'sc' : True, 'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL2
                # {'sc' : False, 'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL2
                # {'sc' : False, 'lr' :  1e-2, 'k' : 1.0, 'maxiter' : 100}, #GardnerKrauthMezard

                # incremental
                #{'incremental': True, 'sc': True},  # Hebbian
                #{'incremental': True, 'sc': True},  # Storkey
                #{'sc' : True, 'lr': 1e-2},  # DOI
                #{'sc' : True, 'lr': 1e-2, 'tol': 1e-1},  # DOII
                #{'sc' : False, 'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.001},  # DescentExpBarrier
                #{'sc' : False, 'incremental': True, 'tol': 1e-1, 'lmbd': 0.5},  # DescentExpBarrierSI
                #{'sc' : False, 'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.001},  # DescentL1
                #{'sc' : False, 'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.001},  # DescentL2
               # {'sc' : True, 'lr' :  1e-2, 'k' : 1.0, 'maxiter' : 100}  # GardnerKrauthMezard
                #,
                # effects of self connectivity
                # {'incremental' : False, 'sc' : False },  #Hebbian
                # {'incremental' : False, 'sc': False },  # Storkey
                # {'sc' : False, 'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL2
                # {'sc' : False, 'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL1
                # {'sc' : False, 'lr': 1e-2, 'k': 1.0, 'maxiter': 100},  # GardnerKrauthMezard
                # {'sc' : False, 'incremental': False, 'tol': 1e-3, 'lmbd': 0.5},  # DescentExpBarrierSI #add bonds
                #{'sc' : False, 'lr': 1e-2, 'tol': 1e-1},  # logistic
                # {'sc' : False, 'lr': 0.05,  'maxiter' : 5000,'goal':[0,0,1,0,0],'symmetric':False} #Infomorphic,redundancy
                #{'sc' : False, 'lr': 0.05,  'maxiter' : 1001,'goal':[0,0,1,0,0],'symmetric':False,'reps':1} #add for reps
                # {'sc' : False, 'lr': 1e-1,  'maxiter' : 1000,'goal':[0,0,-1,0,1],'symmetric':False}, #Infomorphic
                # {'sc' : False, 'lr': 0.05,  'maxiter' : 5000,'goal':[-0.27, -0.68, 0.68, -0.77, -0.8],'symmetric':False}, #optimized (i)
                # # {'sc' : False, 'lr': 0.05,  'maxiter' : 5000,'goal':[0.48, -0.16, 0.25, 0.04, -0.63],'symmetric':False} #optimized (ii)
                # # {'sc' : False, 'lr': 0.05,  'maxiter' : 5000,'goal':[-0.07, -0.31, 0.41, -0.17, -0.65],'symmetric':False} #optimized (iii)
                # {'sc' : False, 'lr': 0.08,  'maxiter' : 40000}
               ]
    for i, rule in enumerate(rules):
        print(rule)
        print('\n')
        params = dict()
        params['rule']  = rule
        params['learning_options'] = options[i]
        params['retrieval_options'] = {'time_of_retrieval' : 50, 'sync' : True}
        #original
        # flips_and_patterns(num_neurons, num_of_flips, num_of_patterns, num_repetitions, params)
        #custom
        # x_range = np.logspace(1,2.3,20,dtype=int)
        x_range = np.linspace(1,2*num_neurons+1,num_points,dtype=int)
        custom_flips_and_patterns(num_neurons,num_of_flips,x_range,num_repetitions,params,seed=seed)

    # for i in range(1,150):
    #     print(i)
    #     weights_distribution_plot(100, i, params)