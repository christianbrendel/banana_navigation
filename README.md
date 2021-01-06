# banana_navigation

### Setup 

To setup this project follow the instructions given here: https://github.com/udacity/deep-reinforcement-learning


### Sturcture 

This project contains three main files: 

- `agent.py` which defines the DQN-agent interacting with the environment.
- `model.py` that contains the definition of the underlying neural network of the agent
- `replay_buffer.py` that contains the class for the replay buffer (which is also part of the agent)

Moreover, to train the model simply exeute the cells in `train.ipynb`. This will train a new agent and saves the model along with a plot of the training history in the folder called model. To test your trained agent run the cells in `test.ipynb`.


### Training Schem)

The current model was trained with default parameters of the the agent (see `agent.py` for details). Moreover we used 

n_episodes = 2000
n_rolling_average = 100
update_every = 4
epsilon = 0.01
