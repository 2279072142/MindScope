import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from collections import deque
import pandas as pd
import json
import time
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def time_function(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 开始时间
        result = function(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 结束时间
        print(f"函数 {function.__name__} 执行耗时: {end_time - start_time} 秒")
        return result

    return wrapper


def objective_function(weights, label, lagent_score, ragent_score):
    lens = len(lagent_score)
    cnt = 0
    for i in range(lens):
        lscore, lbias = get_score(lagent_score[i], weights, 'emma')
        
        rscore, rbias = get_score(ragent_score[i], weights, 'olivia')
        
        if (lscore > rscore):
            cur = lbias
        else:
            cur = rbias
        if cur == label[i].lower():
            cnt += 1
    accuracy = cnt / lens
    return accuracy


def get_score(agent_score, weights, name):
    score = 0
    for key, weight in zip(agent_score[name].keys(), weights):
        score += agent_score[name][key] * weight
    bias = agent_score[name]['cognitive bias name']
    return score, bias


def normalize_to_neg_one_to_one(X):

    X_min, X_max = np.min(X), np.max(X)
    X_norm = 2 * (X - X_min) / (X_max - X_min) - 1
    return X_norm


class Environment:
    def __init__(self, trainset, testset):
        # trainset init
        label = trainset['biasname'].tolist()
        resl = trainset['resl'].tolist()
        resr = trainset['resr'].tolist()
        lagent = [json.loads(score.replace("'", '"').lower()) for score in resl]
        ragent = [json.loads(score.replace("'", '"').lower()) for score in resr]

        # testset init
        testlabel = testset['biasname'].tolist()
        testresl = testset['resl'].tolist()
        testresr = testset['resr'].tolist()
        testlagent = [json.loads(score.replace("'", '"').lower()) for score in testresl]
        testragent = [json.loads(score.replace("'", '"').lower()) for score in testresr]

        #self.parameters = np.random.rand(6) * 2 - 1  # 确保初始化在[-1,1]范围内
        self.parameters = [ 0.5916116 , 0.16982217, -0.42394436 , 0.12357307 , 0.24161538 , 0.29732214]
        self.label = label
        self.lagent = lagent
        self.ragent = ragent
        self.accuracy = 0

        self.testlabel = testlabel
        self.testlagent = testlagent
        self.testragent = testragent

    def step(self, action):
        self.parameters += action
        self.parameters = normalize_to_neg_one_to_one(self.parameters)

        new_accuracy = objective_function(self.parameters, self.label, self.lagent, self.ragent)
        reward = (new_accuracy - self.accuracy) * 5 + (1 - new_accuracy) * 3 
        self.accuracy = new_accuracy  

        test_accuray = objective_function(self.parameters, self.testlabel, self.testlagent, self.testragent)
        return self.parameters, reward, new_accuracy, test_accuray

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Dense(32, input_dim=self.state_size, activation='relu'),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.normal(loc=0, scale=0.1, size=self.action_size)
        else:
            act_values = self.model.predict(state, verbose=0)
            action = act_values[0] + np.random.normal(loc=0, scale=0.05, size=self.action_size)
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

@time_function
def main():
    data_file ="./Data/Debate_data/debate_record.xlsx"
    data = pd.read_excel(data_file)
    traindata = data[:600]
    testdata = data[600:]
    
    env = Environment(traindata, testdata)
    state_size = 6
    action_size = 6
    agent = DQNAgent(state_size, action_size)
    agent.update_target_model()
    batch_size = 2
    C=5
    print(env.parameters)
    
    episodes = 500
    parmeters = []
    bestres = 0
    besttestres = 0
    for e in range(episodes):
        state = np.reshape(env.parameters, [1, state_size])
        for time in range(20):  # replace 20 with max time step
            action = agent.act(state)
            next_state, reward, accuracy, testaccuracy = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % C == 0:
            agent.update_target_model()
        print(f"Episode {e + 1}/{episodes}, reward: {reward}, accuracy:{accuracy},testaccuary:{testaccuracy}")
        print(env.parameters)
        print("")
        if accuracy > bestres:
            bestres = accuracy
            parmeters = env.parameters
            besttestres = testaccuracy
        print(f"best accuracy is {bestres} and testaccuray is {besttestres},best parameters is")
        print(parmeters)

    print(f"best accuracy is {bestres} and testaccuray is {besttestres},best parameters is")
    print(parmeters)



if __name__ == '__main__':
   main()