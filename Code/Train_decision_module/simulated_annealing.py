import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def simulated_annealing(initial_weights, objective_function, max_iterations, initial_temperature, cooling_rate):
    data = pd.read_excel("./Data/Debate_data/debate_record.xlsx")[0:600]
    biasname = data['biasname']
    resl = data['resl']
    resr = data['resr']
    lagent_scores = []
    ragent_scores = []
    for i in range(len(biasname)):
        l_json_data = json.loads(resl[i].replace("'", '"').lower())
        r_json_data = json.loads(resr[i].replace("'", '"').lower())
        lagent_scores.append(l_json_data)
        ragent_scores.append(r_json_data)


    current_solution = initial_weights
    current_cost = objective_function(current_solution,biasname,lagent_scores,ragent_scores)

 
    best_solution = current_solution
    best_cost = current_cost
    best_costs = []
    current_costs = []
  
    temperature = initial_temperature

    for i in range(max_iterations):
       
        new_solution = current_solution + np.random.uniform(-0.1, 0.1, len(current_solution))
        new_solution = np.clip(new_solution, 0, 1)  
        new_solution /= np.sum(new_solution)  
       
        new_cost = objective_function(new_solution,biasname,lagent_scores,ragent_scores)
        
        cost_difference = new_cost - current_cost
        acceptance_probability = np.exp(cost_difference / temperature)
        best_costs.append(best_cost)
        current_costs.append(current_cost)
        
        if cost_difference > 0 or np.random.rand() < acceptance_probability:
            current_solution, current_cost = new_solution, new_cost

            
            if new_cost > best_cost:
                best_solution, best_cost = new_solution, new_cost
        best_costs.append(best_cost)
        current_costs.append(current_cost)
        
        temperature *= cooling_rate

    return  best_solution, best_cost, best_costs, current_costs

def get_score(agent_score,weights,name):
    score=agent_score[name]['argument support'] * weights[0]
    score+=agent_score[name]['logical consistency']*weights[1]
    score+=agent_score[name]['refutation effectiveness']*weights[2]
    score+=agent_score[name]['argument completeness']*weights[3]
    score+=agent_score[name]['persuasiveness']*weights[4]
    score+=agent_score[name]['reasonability assessment of cognitive bias']*weights[5]
    bias = agent_score[name]['cognitive bias name']
    return score,bias

def objective_function(weights,label,lagent_score,ragent_score):

    lens=len(lagent_score)
    cnt=0
    for i in range(lens):
        lscore,lbias=get_score(lagent_score[i],weights,'emma')
        rscore,rbias=get_score(ragent_score[i],weights,'olivia')
        if(lscore>rscore):
            cur=lbias
        else:
            cur=rbias
        if cur==label[i].lower():
            cnt+=1
    accuracy=cnt/lens
    return accuracy
@time_function
def main():
    best_weights=[]
    best_cost=0
    best_cooling_rate = 0
    best_initial_temperature = 0
    initial_weights =np.random.rand(6)
    initial_weights /= initial_weights.sum()
    initial_temperature = 1000
    cooling_rate = 0.100

    for i in tqdm(range(1)):
        for j in range(10):
     
            max_iterations = 100
            cooling_rate=cooling_rate+0.01*i
            initial_temperature=initial_temperature+j*30

            
            optimal_weights, optimal_cost, best_costs, current_costs = simulated_annealing(
                initial_weights,
                objective_function,
                max_iterations,
                initial_temperature,
                cooling_rate
            )
            if optimal_cost>best_cost:
                best_cost=optimal_cost
                best_weights=optimal_weights
                best_cooling_rate=cooling_rate
                best_initial_temperature=initial_temperature
    print("best_weights:", best_weights)
    print("best_cost:", best_cost)
    print("best_initial_temperature",best_initial_temperature)
    print("best_cooling_rate",best_cooling_rate)

if __name__ == '__main__':
    main()