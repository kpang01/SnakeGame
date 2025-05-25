What is Reinforcement Learning?
Reinforcement learning is a machine learning technique used to train an agent (the entity being trained to act upon its environment) using rewards and punishments to teach positive versus negative behavior. For example, in this project, every time the agent eats fruit, it gets a reward. However, every time the agent dies, it receives a punishment. This teaches the agent that getting the fruit is good so it continually becomes better and better.


Breakdown:
All credit goes to Patrick Loeber and his amazing tutorial! This article is essentially a breakdown of everything in the video. The steps are…

Implement the Game and Setup the Environment
Create the Neural Network
Implement and Train the Agent
Now let's get into it!

Step 1: Implement the Game and Setup the Environment
I created the game using python. For this program, you need 4 packages...

NumPy: A python library used for working with arrays
Matplotlib: Helps plot and create visualizations of data
Pytorch: A machine learning framework that helps create neural networks
Pygame: Python module designed for video games
After I downloaded the packages and set up the environment, I implemented the traditional snake game, which is manually controlled by the player. Most of the variables are self-explanatory based on their names, but I will provide a brief explanation below. Be sure to note any important comments!

Setup: Importing necessary items and setting up UI

```python
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80 #ajust the speed of the snake to your liking
# In game.py, this is part of the SnakeGameAI class
# def _update_ui(self):
#         self.display.fill(BLACK)
#         for pt in self.snake:
#             pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
#             pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

#         pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

#         text = font.render("Score: " + str(self.score), True, WHITE)
#         self.display.blit(text, [0, 0])
#         pygame.display.flip()
```

Initializing: Setting up the dimensions of the screen and the state of the game
```python
# In game.py, this is part of the SnakeGameAI class
# def __init__(self, w=640, h=480): #dimensions
#         self.w = w
#         self.h = h
#         self.display = pygame.display.set_mode((self.w, self.h))
#         pygame.display.set_caption('Snake')
#         self.clock = pygame.time.Clock()
#         self.reset()

# def reset(self): #game state
#         self.direction = Direction.RIGHT
#         self.head = Point(self.w / 2, self.h / 2)
#         self.snake = [self.head,
#                       Point(self.head.x - BLOCK_SIZE, self.head.y),
#                       Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
#         self.score = 0
#         self.food = None
#         self._place_food()
#         self.frame_iteration = 0
```
Randomizing Fruit Placement:
```python
# In game.py, this is part of the SnakeGameAI class
# def _place_food(self):
#   x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
#   y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
#   self.food = Point(x, y)
#   if self.food in self.snake:
#         self._place_food()
```
Checking Collisions (aka when the snake dies):
```python
# In game.py, this is part of the SnakeGameAI class
# def is_collision(self, pt=None):
#     if pt is None: #pt is the head of the snake
#         pt = self.head
#     if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
#         return True #if snake hits the side
#     if pt in self.snake[1:]:
#         return True  #if snake hits itself
# return False
```
The State / Actual Playing Process: Implementing reward values and basic game rules
```python
# In game.py, this is part of the SnakeGameAI class
# def play_step(self, action):
#         self.frame_iteration += 1
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 quit()
#         self._move(action)  
#         self.snake.insert(0, self.head)
#         reward = 0
#         game_over = False
#         if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
#             game_over = True
#             reward = -10
#             return reward, game_over, self.score
#         if self.head == self.food:
#             self.score += 1
#             reward = 10
#             self._place_food()
#         else:
#             self.snake.pop()
#         self._update_ui()
#         self.clock.tick(SPEED)
#         return reward, game_over, self.score
```
Setting Up Directions: ensuring the snake can turn in all 4 directions
```python
# In game.py, this is part of the SnakeGameAI class
# def _move(self, action):
#         clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
#         idx = clock_wise.index(self.direction)
#         if np.array_equal(action, [1, 0, 0]): # straight
#             new_dir = clock_wise[idx]  
#         elif np.array_equal(action, [0, 1, 0]): #right turn
#             next_idx = (idx + 1) % 4
#             new_dir = clock_wise[next_idx]  
#         else:  #[0,0,1] aka left turn 
#             next_idx = (idx - 1) % 4
#             new_dir = clock_wise[next_idx]  
#         self.direction = new_dir

#         x = self.head.x
#         y = self.head.y
#         if self.direction == Direction.RIGHT:
#             x += BLOCK_SIZE
#         elif self.direction == Direction.LEFT:
#             x -= BLOCK_SIZE
#         elif self.direction == Direction.DOWN:
#             y += BLOCK_SIZE
#         elif self.direction == Direction.UP:
#             y -= BLOCK_SIZE

#         self.head = Point(x, y)
```
Step 2: Build and Train the Neural Network
After creating the game, I had to create the model to train the network. I had the choice between 2 primary models typically used for reinforcement learning. Q-learning or the Markov Decision Process (MDP). For this game, I decided on deep Q-learning.

Q-Learning:
Q-learning finds the best course of action given the current state of the agent through trial and error. It does so by randomizing its actions, and repeating what works.

Fun Fact: the Q stands for quality!

For example, in the snake game, if the snake repeatedly dies from hitting the walls, at a certain point the agent will learn that going straight towards the wall does NOT lead to the best course of action. Therefore, next time, it will probably turn before it nears a wall.

Instead of a neural network, Q-learning uses a table that calculates the maximum expected future reward for each action at each state. One dimension measures possible actions(the agent’s methods to interact and change its environment), and the other with possible states (the current situation of the agent). Using the table, the agent will choose the action with the highest reward. The table below is a brief example of a Q-learning table with the snake game.
(Image of Q-learning table example would go here)

Q-learning uses the Belleman Equation to create values to make decisions.
(Image of Bellman Equation would go here)

Deep Q-Learning:
Sometimes, a table isn’t very efficient. That’s where deep Q-learning comes in. It is basically the same as Q-learning, except with a neural network instead of a table. It too uses the Belleman equation. This is what I am using to train the snake game!
(Image of Deep Q-Learning diagram would go here)

Luckily, for this specific project, the Belleman and loss function equations can be simplified down to…
(Image of simplified Bellman and loss function equations would go here)

Now let’s get into building the feed-forward neural network!

Importing: In this model, I am using PyTorch
```python
# In model.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import os
```
Building the Neural Network:

Here is a simple feed-forward neural network with an input later of 11 nodes, a single hidden layer of 256 nodes, and an output later of 3 nodes, representing the 3 possible actions the agent can take (right, left, straight).

If you need a recap on neural networks, check out my previous article, Neural Networks for Dummies!
```python
# In model.py
# class Linear_QNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size): #building the input, hidden and output layer
#         super().__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x): #this is a feed-forward neural net
#         x = F.relu(self.linear1(x))
#         x = self.linear2(x)
#         return x

#     def save(self, file_name='model.pth'): #saving the model
#         model_folder_path = './model'
#         if not os.path.exists(model_folder_path):
#             os.makedirs(model_folder_path)

#         file_name = os.path.join(model_folder_path, file_name)
#         torch.save(self.state_dict(), file_name)
```
Training and Optimizing the Network: Here is where I used the simplified deep Q-learning equations mentioned above.
```python
# In model.py
# class QTrainer:
#     def __init__(self, model, lr, gamma): #initializing 
#         self.lr = lr
#         self.gamma = gamma
#         self.model = model
#         self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #optimizer
#         self.criterion = nn.MSELoss() #loss function

#     def train_step(self, state, action, reward, next_state, done): #trainer
#         state = torch.tensor(state, dtype=torch.float)
#         next_state = torch.tensor(next_state, dtype=torch.float)
#         action = torch.tensor(action, dtype=torch.long)
#         reward = torch.tensor(reward, dtype=torch.float)

#         if len(state.shape) == 1: #if there 1 dimension
#             state = torch.unsqueeze(state, 0)
#             next_state = torch.unsqueeze(next_state, 0)
#             action = torch.unsqueeze(action, 0)
#             reward = torch.unsqueeze(reward, 0)
#             done = (done,)
#         pred = self.model(state) #using the Q=model predict equation above

#         target = pred.clone() #using Qnew = r+y(next predicted Q) as mentionned above
#         for idx in range(len(done)):
#             Q_new = reward[idx]
#             if not done[idx]:
#                 Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
#             target[idx][torch.argmax(action[idx]).item()] = Q_new

#         self.optimizer.zero_grad() #calculating loss function
#         loss = self.criterion(target, pred)
#         loss.backward()
#         self.optimizer.step()
```
Graph / Plotting: I implemented a graph to visually communicate the game’s progress while training.
```python
# In helper.py
# import matplotlib.pyplot as plt
# from IPython import display # Note: IPython.display might not work directly in all environments
# plt.ion()
# def plot(scores, mean_scores):
#     display.clear_output(wait=True)
#     display.display(plt.gcf())
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Number of Games')
#     plt.ylabel('Score')
#     plt.plot(scores)
#     plt.plot(mean_scores)
#     plt.ylim(ymin=0)
#     plt.text(len(scores)-1, scores[-1], str(scores[-1]))
#     plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
#     plt.show(block=False)
#     plt.pause(.1)
```
Step 3: Implement the Agent
The last step is to build the agent! This is where all the previous programs come together to create our finished product! After this step, the agent will be self-sufficient enough to play the game on its own, and gradually get better.

Importing and Establishing Parameters:
```python
# In agent.py
# import torch #pytorch
# import random
# import numpy as np #numpy
# from collections import deque #data structure to store memory
# from game import SnakeGameAI, Direction, Point #importing the game created in step 1
# from model import Linear_QNet, QTrainer #importing the neural net from step 2
# from helper import plot #importing the plotter from step 2

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate
```
Initializing: setting up values that will be important later, such as the number of games, the discount rate, memory, and the parameters of the neural network.
```python
# In agent.py, this is part of the Agent class
# def __init__(self):
#         self.n_games = 0
#         self.epsilon = 0  # randomness
#         self.gamma = 0.9  # discount rate
#         self.memory = deque(maxlen=MAX_MEMORY)  
#         self.model = Linear_QNet(11, 256, 3) #input size, hidden size, output size
#         self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
```
Calculating the State: there are points around the head because it determines the state of the snake. The “state” array tells the agent the likeliness of danger or reward based on the direction the snake is headed.
```python
# In agent.py, this is part of the Agent class
# def get_state(self, game):
#         head = game.snake[0]
#         point_l = Point(head.x - 20, head.y)
#         point_r = Point(head.x + 20, head.y)
#         point_u = Point(head.x, head.y - 20)
#         point_d = Point(head.x, head.y + 20)

#         dir_l = game.direction == Direction.LEFT
#         dir_r = game.direction == Direction.RIGHT
#         dir_u = game.direction == Direction.UP
#         dir_d = game.direction == Direction.DOWN

#         state = [
#             (dir_r and game.is_collision(point_r)) or # Danger straight
#             (dir_l and game.is_collision(point_l)) or
#             (dir_u and game.is_collision(point_u)) or
#             (dir_d and game.is_collision(point_d)),

#             (dir_u and game.is_collision(point_r)) or # Danger right
#             (dir_d and game.is_collision(point_l)) or
#             (dir_l and game.is_collision(point_u)) or
#             (dir_r and game.is_collision(point_d)),

#             (dir_d and game.is_collision(point_r)) or # Danger left
#             (dir_u and game.is_collision(point_l)) or
#             (dir_r and game.is_collision(point_u)) or
#             (dir_l and game.is_collision(point_d)),

#             dir_l, #direction
#             dir_r,
#             dir_u,
#             dir_d,

#             game.food.x < game.head.x,  # food left
#             game.food.x > game.head.x,  # food right
#             game.food.y < game.head.y,  # food up
#             game.food.y > game.head.y  # food down
#         ]
#         return np.array(state, dtype=int)
```
Building Memory: this ensures that the agent remembers its training in the long term (over the entire course of time the program remains running) and the short term (the course of time the agent plays a single game).
```python
# In agent.py, this is part of the Agent class
# def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

# def train_long_memory(self):
#         if len(self.memory) > BATCH_SIZE:
#             mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
#         else:
#             mini_sample = self.memory

#         states, actions, rewards, next_states, dones = zip(*mini_sample)
#         self.trainer.train_step(states, actions, rewards, next_states, dones)

# def train_short_memory(self, state, action, reward, next_state, done):
#         self.trainer.train_step(state, action, reward, next_state, done)
```
Having the Agent Take Action: as the agent gets better I want to ensure it is not making decisions based on random moves, but based on what it’s learned from its training.
```python
# In agent.py, this is part of the Agent class
# def get_action(self, state):
#         self.epsilon = 80 - self.n_games # This makes epsilon decrease as n_games increases
#         final_move = [0, 0, 0]
#         if random.randint(0, 200) < self.epsilon: # More random moves at the beginning
#             move = random.randint(0, 2)
#             final_move[move] = 1
#         else: # Less random moves, more model-based decisions later
#             state0 = torch.tensor(state, dtype=torch.float)
#             prediction = self.model(state0)
#             move = torch.argmax(prediction).item()
#             final_move[move] = 1

#         return final_move
```
Training the Agent: making sure the agent repeats the playing process, keeping track of its state, and ensuring the data is being plotted into the graph.
```python
# In agent.py
# def train():
#     plot_scores = []
#     plot_mean_scores = []
#     total_score = 0
#     record = 0
#     agent = Agent() # Make sure Agent class is defined or imported
#     game = SnakeGameAI() # Make sure SnakeGameAI class is defined or imported
#     while True:
#         state_old = agent.get_state(game)
#         final_move = agent.get_action(state_old)
#         reward, done, score = game.play_step(final_move)
#         state_new = agent.get_state(game)
#         agent.train_short_memory(state_old, final_move, reward, state_new, done)
#         agent.remember(state_old, final_move, reward, state_new, done)

#         if done:
#             game.reset()
#             agent.n_games += 1
#             agent.train_long_memory()
#             if score > record:
#                 record = score
#                 agent.model.save()

#             print('Game', agent.n_games, 'Score', score, 'Record:', record)

#             plot_scores.append(score)
#             total_score += score
#             mean_score = total_score / agent.n_games
#             plot_mean_scores.append(mean_score)
#             plot(plot_scores, plot_mean_scores) # Make sure plot function is defined or imported

# if __name__ == '__main__':
#     train()
```
Give yourself a pat on the back because now we’re done!!


Finished Product
After 10 minutes of training, the AI had a high score of 57!
