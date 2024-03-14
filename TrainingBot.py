import retro    # import retro to play Street Fighter using a ROM

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
    
    def step(self, action):
        # Take a step
        obs, reward, done, info = self.game.step(action)
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
        self.game.render()


    def reset(self):    
        # Return the first frame
        obs = self.game.reset()
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

# Testing
env = StreetFighter()
env.observation_space.sample()  # sample the observation space
# env.action_space.sample()   # samples the action space
print (env.observation_space.shape)
print (env.action_space.shape)
obs = env.reset()   # resets game to starting state
done = False        # Set flag to false
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()    # renders environment
        obs, reward, done, info = env.step(env.action_space.sample())   # randomly takes action
        time.sleep(0.05)
        if reward > 0:
            print(reward)
env.close()

# plt.imshow(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)) # Shows the frame data (what has changed)
