import numpy as np

#Will follow an epsilon greedy policy, given Q to determine the next action 
#Meaning it will determine the best action to take in the current moment
def genAction(state, e, Q):
    probHit = Q[state][1]
    probStick = Q[state][0]
    
    if probHit>probStick:
        probs = [e, 1-e]
    elif probStick>probHit:
        probs = [1-e, e]
    else:
        probs = [0.5, 0.5]
        
    action = np.random.choice(np.arange(2), p=probs)   
    return action


#Will change Q after each completed game/episode
#This is where the "learning" is taking place
def setQ(Q, currentEpisode, gamma, alpha):
    for t in range(len(currentEpisode)):
        #episode[t+1:,2] gives all the rewards in the episode from t+1 onwords
        rewards = currentEpisode[t:,2]
        #Create a list with the gamma rate increasing
        discountRate = [gamma**i for i in range(1,len(rewards)+1)]
        #Discounting the rewards from t+1 onwards
        updatedReward = rewards*discountRate
        #Summing up the discounted rewards to equal the return at time step t
        Gt = np.sum(updatedReward)
        #Calculating the actual Q table value of the state, actionn pair. 
        Q[currentEpisode[t][0]][currentEpisode[t][1]] += alpha *(Gt - Q[currentEpisode[t][0]][currentEpisode[t][1]])
    return Q







    
