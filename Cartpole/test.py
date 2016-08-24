import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
from os.path import isfile

num_inputs = 4
Weight_File = "model.h5"
env = gym.make('CartPole-v0')

# load json and create model
if isfile(Weight_File):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
else:
    print ("Weight file doesn't exist!")

if __name__ == '__main__':
    numEpisodes = 2000
    numRuns = 10
    for i_episode in range(numEpisodes):
        current_state = env.reset()
        t = 0
        returns = 0
        while(1):
            env.render()
            scores = loaded_model.predict(current_state.reshape(1,num_inputs))
            action = scores.argmax()
            #print(current_state,type(current_state),env.action_space.sample())
            next_state, reward, done, info = env.step(action)
            returns += reward
            t += 1
            if done or returns>200:
                print("Episode " + str(i_episode) + " finished after {} timesteps with {} reward".format(t+1,returns))
                break



