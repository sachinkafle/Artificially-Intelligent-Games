import gym

from tensorflow.keras.models import Sequential

#layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import Adam

from collections import deque

import random
import numpy as np


env = gym.make("MsPacman-v0")

env.reset() #buffer clear

state_size = (88, 80, 1)

action_size = env.action_space.n #9

#deep q learning model: Deep convolution Q-learning

color = np.array([210, 164, 74]).mean()

def preprocess_state(state):

    #crop and resize the image
    image = state[1:176:2, ::2]

    #convert the image to greyscale
    image = image.mean(axis = 2)

    #improve image contrast
    image[image == color] = 0 #to find patterns during convolution layer

    #normalize the image
    image = (image - 128) / 128 - 1 
    
    #reshape the image
    image = np.expand_dims(image.reshape(88, 80, 1), axis = 0)

    return image
    

#Deep Q Learning / Deep Convolution Q
class DQN:
	def __init__(self, state_size, action_size):
		#define state size and action size
		self.state_size = state_size
		self.action_size = action_size
		
		#epison values
		self.epsilon = 0.8 #80 % -> random , 20% choose laargest Q values
		
		self.gamma = 0.2 #discount factor
		
		self.update_rate = 4000
		
		self.replay_buffer = deque(maxlen = 5000) #data is used for training
		
		self.main_network = self.build_network() #prediction of the Q values 
		
		self.target_network = self.build_network() # Q -values of target (Actual Q value)
		
		# assign theta to theta dash 
		
		self.target_network.set_weights(self.main_network.get_weights())
		
		
	def build_network(self):
		model = Sequential()
		
		#layer 1
		model.add(Conv2D(32, (8, 8), strides = 4, padding = "same", input_shape = self.state_size))  #features from the image
		model.add(Activation('relu')) #we are going to remove noise
		#-----------------------------------------------
		
		
		
		#layer 2
		model.add(Conv2D(64, (4,4), strides = 2, padding = "same"))
		model.add(Activation('relu'))
		
		
		#layer 3:
		model.add(Conv2D(64, (3,3), strides = 1, padding = "same"))
		model.add(Activation('relu'))
		
		#flatten layer: input layer
		model.add(Flatten()) #flattening of data is done before sending to FC - neural network
		
		#hidden layer
		model.add(Dense(512, activation  = "relu"))
		
		#output layer: 4 Q values: 4 actions
		
		#pacman: 9 actions, 9 neurons, depict Q values
		model.add(Dense(self.action_size, activation = "linear"))
		
		model.compile(loss = "mse", optimizer = Adam()) #smooth training of the data
		
		return model
		
		
	def store_transition(self, current_state, action, reward, next_state, done ):
		self.replay_buffer.append((current_state, action, reward, next_state, done))
		
		
	#We learned that in DQN, to take care of exploration-exploitation trade off, we select action
    #using the epsilon-greedy policy. So, now we define the function called epsilon_greedy
    #for selecting action using the epsilon-greedy policy.	
		
	def epsilon_greedy(self, state):
		if random.uniform(0, 1) < self.epsilon:
			return np.random.randint(self.action_size)
		Q_values = self.main_network.predict(state) 
		
		return np.argmax(Q_values[0]) #action that leads to the maximum Q value
		
			
			
	#Two neural network : Main Network, Target Network
	
	def train(self, batchsize):
		minibatch = random.sample(self.replay_buffer, batch_size)
		
		#compute the Q-value
		
		#Target network: optimal Q value : Actual Q value
		for state, action, reward, next_state, done in minibatch:
			if not done:
				target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
			else:
				target_Q = reward 
				
		#target_Q -> optimal Q value : best
		
		
		#main network: Predicted Q value : might be bad
		Q_values = self.main_network.predict(state)
		
		Q_values[0][action] = target_Q
		
		#train our model
		self.main_network.fit(state, Q_values, epochs = 1, verbose = 0) #training/backpropagation -> theta or weights will change
		

num_episodes = 500
num_timesteps = 20000 #iteration



batch_size = 8 
num_screens = 4

dqn = DQN(state_size, action_size)

done = False

time_step = 0


#for each episode
for i in range(num_episodes):
    
    #set return to 0
    Return = 0 #sum of rewards in that episode
    
    #preprocess the game screen
    state = preprocess_state(env.reset()) #buffer clearance

    #for each step in the episode
    for t in range(num_timesteps):
        env.render()
        
        #update the time step
        time_step = time_step + 1
        
        #update the target network
        
        #in every 4000 time steps update the target network
        if time_step % dqn.update_rate == 0:
        	self.target_network.set_weights(self.main_network.get_weights())
        
        #select the action
        action = dqn.epsilon_greedy(state)
        
        #perform the selected action
        next_state, reward, done, _  = env.step(action) #captcha : google 
        
        #preprocess the next state
        next_state = preprocess_state(next_state) 
        
        #store the transition information
        dqn.store_transition(state, action, reward, next_state, done)
        
        
        #update current state to next state
        state = next_state
        
        #update the return
        Return= Return + reward
        
        #if the episode is done then print the return
        if done:
        	print(f'Episode: {i} -> Return: {Return}')
        	break
        
            
        #if the number of transitions in the replay buffer is greater than batch size
        #then train the network
        #if we got new batch 
        if len(dqn.replay_buffer) > batch_size:
        	dqn.train(batch_size)

		
		
		
		





