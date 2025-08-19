from environment import Environment
import numpy as np

env = Environment()
rewards = env.rewardBoard
Q_Table = rewards.copy()


class Q_Agent():
    # Intialise
    def __init__(self,alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def prepare_states(self, rewards):
        self.possibleStates = list()
        for i in range(rewards.shape[0]):
            if sum(abs(rewards[i])) != 0:
                  self.possibleStates.append(i) 
        return self.possibleStates    

    def return_max(self, qvalues):
        ind = 0
        maxQValue = -np.inf
        for i in range(len(qvalues)):
            if (qvalues[i] > maxQValue) and (qvalues[i] != 0):
                maxQValue = qvalues[i]
                ind = i
        return ind, maxQValue

    def possibleActions(self, rewards, currentPos):
        self.validActions = list()
        for i in range(rewards.shape[1]):
            if (rewards[currentPos][i] != 0):
                self.validActions.append(i)
        return self.validActions
        
	
    
    def learn(self, reward, maxQValue, startingPos, action):
        
        TD = reward + self.gamma * maxQValue - Q_Table[startingPos][action]
        
        
        Q_Table[startingPos][action] += self.alpha*TD


qa = Q_Agent(alpha = 0.75, gamma = 0.9)

for epoch in range(10000):
	startingPos = np.random.choice(qa.prepare_states(rewards))
	
	action = np.random.choice(qa.possibleActions(rewards, startingPos))
	
	reward = rewards[startingPos][action]
	
	_, maxQValue = qa.return_max(Q_Table[action])
	qa.learn(reward, maxQValue, startingPos, action)

currentPos = env.startingPos
while True:
	action, _ = qa.return_max(Q_Table[currentPos])
	env.movePlayer(action)
	currentPos = action
