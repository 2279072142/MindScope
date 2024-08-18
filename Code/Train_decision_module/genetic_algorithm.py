import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import itertools

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

def initialize_population(pop_size, gene_length):
    return np.random.rand(pop_size, gene_length)

def evaluate_population(population, label, lagent_score, ragent_score):
    scores = []
    for individual in population:
        normalized_weights = individual / np.sum(individual)
        score = objective_function(normalized_weights, label, lagent_score, ragent_score)
        scores.append(score)
    return scores

def select(population, scores, num_parents):
    parents = np.zeros((num_parents, population.shape[1]))
    for parent_num in range(num_parents):
        max_score_idx = np.argmax(scores)
        parents[parent_num, :] = population[max_score_idx, :]
        scores[max_score_idx] = -99999999  # Avoid selecting the same individual again
    return parents

def crossover(parents, offspring_size, crossover_rate):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        # Applying crossover_rate
        if np.random.rand() < crossover_rate:
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            crossover_point = np.random.randint(1, offspring_size[1]-1)
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        else:
            offspring[k, :] = parents[k % parents.shape[0], :]
    return offspring

def mutation(offspring_crossover, mutation_rate):
    for idx in range(offspring_crossover.shape[0]):
        for gene in range(offspring_crossover.shape[1]):
            if np.random.rand() < mutation_rate:
                # Adding a random value to the gene
                random_value = np.random.uniform(-0.1, 0.1)
                offspring_crossover[idx, gene] = offspring_crossover[idx, gene] + random_value
    return offspring_crossover

def genetic_algorithm(data_file, pop_size, max_generations, crossover_rate, mutation_rate):
    # Loading data
    data = pd.read_excel(data_file)
    biasname = data['biasname'].tolist()
    resl = data['resl'].tolist()
    resr = data['resr'].tolist()
    lagent_scores = [json.loads(score.replace("'", '"').lower()) for score in resl]
    ragent_scores = [json.loads(score.replace("'", '"').lower()) for score in resr]

    gene_length = 6
    population = initialize_population(pop_size, gene_length)

    best_score = -np.inf
    best_weights = None

    for generation in range(max_generations):
        scores = evaluate_population(population, biasname, lagent_scores, ragent_scores)
        parents = select(population, scores, int(pop_size / 2))
        offspring_crossover = crossover(parents, (pop_size - parents.shape[0], gene_length), crossover_rate)
        offspring_mutation = mutation(offspring_crossover, mutation_rate)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
        current_best_score = np.max(scores)
        if current_best_score > best_score:
            best_score = current_best_score
            best_weights = population[np.argmax(scores), :] / np.sum(population[np.argmax(scores), :])

    return best_score, best_weights

def search_genetic_algorithm_params(data_file, parameter_grid):
    best_score = -np.inf
    best_params = None
    best_weights = None

    for params in tqdm(list(itertools.product(*parameter_grid.values())), desc="Searching Parameters"):
        pop_size, max_generations, crossover_rate, mutation_rate = params
        score, weights = genetic_algorithm(data_file, pop_size, max_generations, crossover_rate, mutation_rate)
        if score > best_score:
            best_score = score
            best_params = {'pop_size': pop_size, 'max_generations': max_generations, 'crossover_rate': crossover_rate, 'mutation_rate': mutation_rate}
            best_weights = weights

    return best_score, best_params, best_weights


def test_step(data_file,weight):
    data=pd.read_excel(data_file)[600:]
    biasname = data['biasname'].tolist()
    resl = data['resl'].tolist()
    resr = data['resr'].tolist()
    lagent_scores = [json.loads(score.replace("'", '"').lower()) for score in resl]
    ragent_scores = [json.loads(score.replace("'", '"').lower()) for score in resr]
    score=objective_function(weight,biasname,lagent_scores,ragent_scores)
    print(f"the accuracy is {score}!")

if __name__ == '__main__':
    data_file = "./Data/Debate_data/debate_record.xlsx"
    parameter_grid = {
        'pop_size': range(20, 30,5),
        'max_generations': [200],
        'crossover_rate': [0.1,0.2], #[0.5,0.6,0.7,0.8,0.9]
        'mutation_rate': [0.01,0.03,0.05] #[,0.12,0.14,0.16,0.18,0.2]
    }

    best_score, best_params, best_weights = search_genetic_algorithm_params(data_file, parameter_grid)
    print("Best Score:", best_score)
    print("Best Parameters:", best_params)
    print("Best Weights:", best_weights)
