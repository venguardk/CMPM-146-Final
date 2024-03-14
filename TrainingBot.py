import retro    # import retro to play Street Fighter using a ROM


retro.data.list_games() # list the games imported
# python -m retro.import . # run this from the roms folder

# creates the game enviornment
env = retro.make(game = 'StreetFighterIISpecialChampionEdition-Genesis')
# env.close() # closes enviornment

env.observation_space.sample()  # sample the observation space
env.action_space.sample()   # samples the action space

# Testing environment
obs = env.reset()   # resets game to starting state
done = False        # Set flag to false
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()    # renders environment
        obs, reward, done, info = env.step(env.action_space.sample())   # randomly takes action
        print(reward)