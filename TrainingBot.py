import retro    # import retro to play Street Fighter using a ROM
import pygame
import optuna   # Importing optimization frame
# PPO alg for RL
from stable_baselines3 import PPO
# Bring in the eval policy method for metric calculation
from stable_baselines3.common.evaluation import evaluate_policy
# Import the sb3 monitor for logging
from stable_baselines3.common.monitor import Monitor
# Import the vec wrappers to Vectorize and frame stack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import os

# python -m retro.import . # run this from the roms folder

# # creates the game enviornment
# env = retro.make(game = 'StreetFighterIISpecialChampionEdition-Genesis')
# env.close() # closes enviornment

# # checking mb delete later
# env.observation_space.sample()  # sample the observation space
# env.action_space.sample()   # samples the action space

# # Testing environment
# obs = env.reset()   # resets game to starting state
# done = False        # Set flag to false
# for game in range(1):
#     while not done:
#         if done:
#             obs = env.reset()
#         env.render()    # renders environment
#         obs, reward, done, info = env.step(env.action_space.sample())   # randomly takes action
#         print(reward)
# env.close()

# Setup Environment
"""
    - Observation Preprocess - grayscale [Done], frame delta, resize the frame [Done]
    - Filter the action - parameter [Done]
    - Reward Function - set this to score [Subject to change]
""" 

#pip install opencv-python

# imports environment base calss for a wrapper
from gym import Env     
# Import the space shapes for the environment
from gym.spaces import MultiBinary, Box
# Import numpy to calculate frame delta
import numpy as np
# Import opencv for grayscaling
import cv2
# Import matplotlib for plotting the image
from matplotlib import pyplot as plt
# Importing time
import time

"""
    1. Frame
    2. Preprocess 200x256x3 > 84x84x1
    3. change in pixels current_frame - the last frame
"""

pygame.init()
win = pygame.display.set_mode((800, 600))

buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

action_array = [0,0,0,0,0,0,0,0,0,0,0,0]


# Create custom environment
class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        # Specify action space and observation space
        self.observation_space = Box(low= 0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)   # action space
        # Startup and instance of the game
        self.game = retro.make(game = 'StreetFighterIISpecialChampionEdition-Genesis', 
                               use_restricted_actions=retro.Actions.FILTERED)
        self.raw_obs = np.zeros((200,256,3), dtype=np.uint8)
    
    def step(self, action):
        # Take a step
        obs, reward, done, info = self.game.step(action)
        self.raw_obs = obs
        obs = self.preprocess(obs)

        # Frame delta
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        # Reshape the reward function
        # v Note WILL NEED TO CHANGE FOR OUR MODEL v
        reward = info['score'] - self.score
        self.score = info['score']
        
        return frame_delta, reward, done, info

    def render(self, *args, **kwargs):
        img = pygame.image.frombuffer(self.raw_obs.tobytes(), (self.raw_obs.shape[1], self.raw_obs.shape[0]), 'RGB')
        img = pygame.transform.scale(img, (800, 600))
        win.blit(img, (0, 0)) # commented out when gameplay loop was commented out 
        pygame.display.flip()


    def reset(self):    
        # Return the first frame
        obs = self.game.reset()
        self.raw_obs = obs
        obs = self.preprocess(obs)
        # current frame - previous frame
        self.previous_frame = obs

        #  Create an attribute to hold the score delta
        self.score = 0      
        return obs
        
    
    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)    # Grayscaling
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC) # Resize
        channels = np.reshape(resize, (84, 84, 1))  # Add the channels value
        return channels


    def close(self):
        self.game.close()

# Log directory
LOG_DIR = './logs/'
OPT_DIR = './opt/'

# Function to return test hyperparameters - define the object function
def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }

# Run a training Loop and return mean reward
best_reward = -float('inf')
best_model = str()
def optimize_agent(trial):
    global best_model, best_reward
    try:
        model_params = optimize_ppo(trial)
        
        # Create environment
        env = StreetFighter()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        # Create algo
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params) # important
        model.learn(total_timesteps=100000) # change for training speed

        #Evaluate model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)
        env.close()

        # Saving model
        """
            # to keep track of the best model:
            compare mean_reward to best_reward - a global variable
                if greater, saves model to 'best_model'.format
            save it the same way as we did before
            
        """
        if mean_reward > best_reward:
            best_model = 'trial_{}_best_model'.format(trial.number)
            print(best_model)
            best_reward = mean_reward
        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)  

        return mean_reward
    except Exception as e:
        return -1000

# # Creating the study
study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=10, n_jobs=1)  
study.best_params
study.best_trials

model = PPO.load(os.path.join(OPT_DIR, best_model))

# Setup Callback-----------------------------------------------------------------------------------
# Import base callback
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    
CHECKPOINT_DIR = './train/'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# fine tuning the model ---------------------------------------------------------------------------
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, 4, channels_order='last')

model_params = study.best_params
# model_params['n_steps'] = 7488  # set n_steps to 7488 or a factor of 64
model_params = {'n_steps': 2570.949, 'gamma': 0.906, 'learning_rate': 2e-07, 'clip_range': 0.369, 'gae_lambda': 0.891}
# model_params['Learning_rate'] = 5e-7
model_params['n_steps'] = 40*64
model_params

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
# Reload previous weights from HPO for transfer learning
model.load(os.path.join(OPT_DIR, best_model))
# Kick off training
model.learn(total_timesteps = 5000000, callback=callback)
# model.learn(total_timestep = 5000000)

# Testing the Model---------------------------------------------------------------------------------
# Evaluating
model = PPO.load('./opt/' + best_model)
mean_reward,_ = evaluate_policy(model, env, render=True, n_eval_episodes=5)

# Testing loop
obs = env.reset()
obs.shape
env.step(model.predict(obs)[0])
obs = env.reset()
done = False
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()
        action = model.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        time.sleep(0.01)
        if (reward != 0):
            print(reward)
env.close()

# Gameplay Loop-------------------------------------------------------------------------------------
env = StreetFighter()
#env.observation_space.sample()  # sample the observation space
# env.action_space.sample()   # samples the action space
#print (env.observation_space.shape)
#print (env.action_space.shape)
obs = env.reset()   # resets game to starting state

#pygame.init()
#win = pygame.display.set_mode((800, 600))

#buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

#action_array = [0,0,0,0,0,0,0,0,0,0,0,0]

done = False        # Set flag to false
for game in range(1):
    while not done:
        if done:
            obs = env.reset()

        actions = set()

        # Display
        env.render()

        # Control Events
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            # movement
            if keys[pygame.K_d]:
                actions.add('RIGHT')
            if keys[pygame.K_a]:
                actions.add('LEFT')
            if keys[pygame.K_s]:
                actions.add('DOWN')
            if keys[pygame.K_w]:
                actions.add('UP')
            # kicks
            if keys[pygame.K_j]:
                actions.add('A')
            if keys[pygame.K_k]:
                actions.add('B')
            if keys[pygame.K_l]:
                actions.add('C')
            # punches
            if keys[pygame.K_u]:
                actions.add('X')
            if keys[pygame.K_i]:
                actions.add('Y')
            if keys[pygame.K_o]:
                actions.add('Z')

            for i, a in enumerate(buttons):
                if a in actions:
                    action_array[i] = 1
                else:
                    action_array[i] = 0

        # taking action
        # obs, reward, done, info = env.step(action_array)   # for human input as player 1
        obs, reward, done, info = env.step(env.action_space.sample())   # for testing w/ random AI
        # time.sleep(0.01)

env.close()

plt.imshow(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)) # Shows the frame data (what has changed)


"""
    So, we want to be able to train a base model to use to further train, after a match with a player, right?
        How do we do this?
        
"""