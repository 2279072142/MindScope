import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization,LeakyReLU
from keras.optimizers import Adam
from collections import deque
import pandas as pd
import json
import time
import tensorflow as tf
import os

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
    def __init__(self,label,lagent,ragent):
        #self.parameters = np.random.rand(6) * 2 - 1  # 确保初始化在[-1,1]范围内
        self.parameters =[15.49208602,13.79171383, 0.81506395,-7.07399149 ,14.35630417 ,-1.61797024]
        self.label=label
        self.lagent=lagent
        self.ragent=ragent
        self.accuracy=0


    def step(self, action):
        self.parameters += action
        self.parameters = normalize_to_neg_one_to_one(self.parameters)
   
        new_accuracy = objective_function(self.parameters, self.label, self.lagent, self.ragent)
        reward = (new_accuracy-self.accuracy)*2+(1-new_accuracy)*5 
        self.accuracy = new_accuracy 

        #reward = max(0, reward)  
        return self.parameters, reward,new_accuracy

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.80  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.002
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())  
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5)) 
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))


    def act(self, state):
        if np.random.rand() <= self.epsilon:

            action = np.random.rand(self.action_size) * 2 - 1  
        else:
            act_values = self.model.predict(state, verbose=0)
            return act_values[0]  

        return action  

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(next_state,verbose=0)[0])
            target_f = self.model.predict(state,verbose=0)
            target_f[0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    data_file = "./Data/Debate_data/debate_record.xlsx"
    data = pd.read_excel(data_file)
    biasname = data['biasname'].tolist()
    resl = data['resl'].tolist()
    resr = data['resr'].tolist()
    lagent_scores = [json.loads(score.replace("'", '"').lower()) for score in resl]
    ragent_scores = [json.loads(score.replace("'", '"').lower()) for score in resr]


    env = Environment(biasname,lagent_scores,ragent_scores)
    state_size = 6
    action_size = 6
    agent = DQNAgent(state_size, action_size)
    batch_size = 2
    print(env.parameters)

    episodes = 200
    parmeters=[]
    bestres=0
    for e in range(episodes):
        state = np.reshape(env.parameters, [1, state_size])
        for time in range(30):
            action = agent.act(state)
            next_state, reward, accuracy = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"Episode {e + 1}/{episodes}, reward: {reward}, accuracy:{accuracy}")
        print(env.parameters)
        if accuracy>bestres:
            bestres=accuracy
            parmeters=env.parameters
    print(f"best accuracy is {bestres},best parameters is")
    print(parmeters)

if __name__ == '__main__':
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    main()


