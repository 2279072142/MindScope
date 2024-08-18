import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import itertools


import time
alltime=0

def get_score(agent_score, weights, name):
    score = 0
    for key, weight in zip(agent_score[name].keys(), weights):
        score += agent_score[name][key] * weight
    bias = agent_score[name]['cognitive bias name']
    return score, bias

def objective_function(weights, label, lagent_score, ragent_score):
    lens = len(lagent_score)
    cnt = 0
    for i in range(lens):
        lscore, lbias = get_score(lagent_score[i], weights, 'emma')
        rscore, rbias = get_score(ragent_score[i], weights, 'olivia')
        if(lscore > rscore):
            cur = lbias
        else:
            cur = rbias
        if cur == label[i].lower():
            cnt += 1
    accuracy = cnt / lens
    return accuracy


def initialize_pheromone_matrix(num_weights, initial_pheromone):
    return np.full((num_weights, num_weights), initial_pheromone)


def update_pheromone(pheromone_matrix, ants_solutions, decay_rate, scores):
    for i, solution in enumerate(ants_solutions):
        for j in range(len(solution) - 1):
            pheromone_matrix[solution[j], solution[j+1]] += scores[i]
    pheromone_matrix = pheromone_matrix * (1 - decay_rate)
    return pheromone_matrix

def generate_new_solution(pheromone_matrix, alpha, beta):
    num_weights = pheromone_matrix.shape[0]
    solution = np.random.permutation(num_weights)
    return solution



def ant_colony_optimization(data_file, num_ants, num_generations, alpha, beta, decay_rate, initial_pheromone):

    data = pd.read_excel(data_file)
    biasname = data['biasname'].tolist() 
    resl = data['resl'].tolist()  
    resr = data['resr'].tolist()  
    lagent_scores = [json.loads(score.replace("'", '"').lower()) for score in resl]
    ragent_scores = [json.loads(score.replace("'", '"').lower()) for score in resr]

    num_weights = 6  
    pheromone_matrix = initialize_pheromone_matrix(num_weights, initial_pheromone)

    best_score = -np.inf  
    best_solution_normalized = np.zeros(num_weights)  

    for generation in tqdm(range(num_generations)):
        ants_solutions = [generate_new_solution(pheromone_matrix, alpha, beta) for _ in range(num_ants)]
        scores = [objective_function(solution / np.sum(solution), biasname, lagent_scores, ragent_scores) for solution in ants_solutions]
        pheromone_matrix = update_pheromone(pheromone_matrix, ants_solutions, decay_rate, scores)

     
        current_best_score_index = np.argmax(scores)
        if scores[current_best_score_index] > best_score:
            best_score = scores[current_best_score_index]
            best_solution_normalized = ants_solutions[current_best_score_index] / np.sum(ants_solutions[current_best_score_index])

    return best_score, best_solution_normalized  




def grid_search_aco(data_file, parameter_grid,istest=False):
    if istest:
        num_ants, num_generations, alpha, beta, decay_rate, initial_pheromone = [5,100,1.0,1.0,0.6,0.1]

        
        current_score, current_weights = ant_colony_optimization(data_file, num_ants, num_generations, alpha, beta,
                                                                 decay_rate,
                                                                 initial_pheromone)
        print(current_score)
        print(current_weights)
        return current_score,None,current_weights


   
    best_score = -np.inf
    best_params = None
    best_weights= []

    
    for params in tqdm(list(itertools.product(*parameter_grid.values()))):
        
        num_ants, num_generations, alpha, beta, decay_rate, initial_pheromone = params

        
        current_score,current_weights = ant_colony_optimization(data_file, num_ants, num_generations, alpha, beta, decay_rate,
                                                initial_pheromone)

        
        if current_score > best_score:
            best_score = current_score
            best_weights=current_weights
            best_params = {
                'num_ants': num_ants,
                'num_generations': num_generations,
                'alpha': alpha,
                'beta': beta,
                'decay_rate': decay_rate,
                'initial_pheromone': initial_pheromone
            }

    return best_score, best_params,best_weights


parameter_grid = {
    'num_ants': [5, 10, 15,20,25],
    'num_generations': [50, 100,200],
    'alpha': [1.0, 2.0],
    'beta': [1.0, 2.0],
    'decay_rate': [0.2,0.4, 0.6, 0.8],
    'initial_pheromone': [0.1, 0.01]
}



best_score, best_params, best_weights = grid_search_aco("./Data/Debate_data/debate_record.xlsx", parameter_grid,
                                                            True)

print("最佳分数:", best_score)
print("最佳参数:", best_params)
print("最佳权重",best_weights)
