import os
import gym
import pylab
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class A2CAgent:
    
    def __init__(self, state_size, action_size, load_models = False, actor_model_file = "", critic_model_file = ""):
              
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # Hyper parameters for learning
        self.discount_factor = 0.99
        self.actor_learning_rate = 0.0005
        self.critic_learning_rate = 0.005

        # Create actor and critic neural networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if load_models:
            if actor_model_file:
                self.actor.load_weights(actor_model_file)
            if critic_model_file:
                self.critic.load_weights(critic_model_file)

    # The actor takes a state and outputs probabilities of each possible action
    def build_actor(self):
        
        layer1 = Dense(16, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform')
        layer2 = Dense(16, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform')
        # Use softmax activation so that the sum of probabilities of the actions becomes 1
        layer3 = Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform')
        
        actor = Sequential(layers = [layer1, layer2, layer3]) 
        
        # Print a summary of the network
        actor.summary()
        
        # We use categorical crossentropy loss since we have a probability distribution
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_learning_rate))
        return actor

    # The critic takes a state and outputs the predicted value of the state
    def build_critic(self):
        
        layer1 = Dense(16, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform')
        layer2 = Dense(16, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform')
        layer3 = Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform')
        
        critic = Sequential(layers = [layer1, layer2, layer3])
        
        # Print a summary of the network
        critic.summary()
        
        critic.compile(loss="mean_squared_error", optimizer=Adam(lr=self.critic_learning_rate))
        return critic

    # Randomly pick an action, with probabilities from the actor network
    def get_action(self, state):
        # Get probabilities for each action
        policy = self.actor.predict(state, batch_size=1).flatten()
        # Randomly choose an action
        return np.random.choice(self.action_size, 1, p=policy).take(0)

    def train_model(self, previous_state, action, reward, current_state, done):

        # Make predictions of the value using the critic
        predicted_value_previous_state = self.critic.predict(previous_state)[0]
        predicted_value_current_state = self.critic.predict(current_state)[0] if not done else 0.
        
        # Estimate the "real" value as the reward + the (discounted) predicted value of the current state 
        real_previous_value = reward + self.discount_factor * predicted_value_current_state
        
        advantages = np.zeros((1, self.action_size))
        # The advantage is the difference between what we got and what we predicted
        # - put it in the "slot" of the current action
        advantages[0][action] = real_previous_value - predicted_value_previous_state
        
        # Train the actor and the critic
        self.actor.fit(previous_state, advantages, epochs=1, verbose=0)
        self.critic.fit(previous_state, reshape(real_previous_value), epochs=1, verbose=0)


# Reshape array for input into keras
def reshape(state):
    return np.reshape(state, (1, -1))


SAVE_DIRECTORY = "./a2c_output/"
ACTOR_MODEL_FILE = SAVE_DIRECTORY + "actor.h5"
CRITIC_MODEL_FILE = SAVE_DIRECTORY + "critic.h5"
LOAD_MODELS = False
GYM_GAME = "CartPole-v1"


def runSavedModel():
    env = gym.make(GYM_GAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A2CAgent(state_size, action_size, True, ACTOR_MODEL_FILE, CRITIC_MODEL_FILE)
    try:
        while(True):
            state = reshape(env.reset())
            done = False
            while not done:
                env.render()
                action = agent.get_action(state)
                state, reward, done, info = env.step(action)
                state = reshape(state)
    except:
        env.close()


def trainModel():
    RENDER = False
    FAIL_PENALTY = 100
    SUCCESS_COUNT = 10 # Number of max scores in a row that constitutes success
    
    if not os.path.exists(SAVE_DIRECTORY):
        os.mkdir(SAVE_DIRECTORY)

    env = gym.make(GYM_GAME)
    # In CartPole the maximum length of an episode is 500
    max_score = 500
    
    # Get sizes of state and action from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size, LOAD_MODELS, ACTOR_MODEL_FILE, CRITIC_MODEL_FILE)

    scores, episodes = [], []
    
    episode = 0
    all_done = False
 
    while not all_done:
        env_done = False
        score = 0
        state = reshape(env.reset())

        while not env_done:
            if RENDER:
                env.render()

            action = agent.get_action(state)
            next_state, reward, env_done, info = env.step(action)
            next_state = reshape(next_state)
            
            score += reward
            
            # If an action makes the episode end, give it a big penalty
            # but reduce the penalty in proportion to the score so far
            reward = reward if not env_done else (score * (FAIL_PENALTY/max_score) - FAIL_PENALTY)

            agent.train_model(state, action, reward, next_state, env_done)
           
            state = next_state

            if env_done:
                score = score if score == max_score else score
                scores.append(score)
                episodes.append(episode)
                
                pylab.plot(episodes, scores, 'b')
                pylab.pause(0.1)
                                
                print("episode:", episode, " score:", score)

                # If we have SUCCESS_COUNT perfect runs in a row, stop training
                if len(scores) >= SUCCESS_COUNT and np.mean(scores[-SUCCESS_COUNT:]) == max_score:
                    all_done = True
        
        episode = episode + 1      
        
        # Save the model
        if all_done:
            agent.actor.save_weights(ACTOR_MODEL_FILE)
            agent.critic.save_weights(CRITIC_MODEL_FILE)

    env.close()


#if __name__ == "__main__":
#    trainModel()
    