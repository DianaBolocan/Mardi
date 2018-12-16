from ple.games.snake import Snake
from ple import PLE
import pygame
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import math

class MyAgent:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(9, input_dim=4, activation="relu"))
        self.model.add(Dense(15, activation="relu"))
        self.model.add(Dense(3))
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(sgd, "mse")

    def train(self, train_set, target_set):
        self.model.fit(x=train_set, y=target_set, epochs=1, verbose=0)

    def getAction(self, state):
        return self.model.predict(state.reshape(1, 4))

    def train_batch(self, train_set):
        y = 0.95
        target_set = []
        for current_state, action, reward, new_state in train_set:
            target = reward + y * np.max(agent.getAction(new_state))
            target_vector = agent.getAction(current_state)[0]
            target_vector[action] = target
            target_set.append(target_vector)
        train_set = [i[0] for i in train_set]
        self.train(np.array(train_set), np.array(target_set))


# get action from currently_available_actions + index:
def nextAction(agent,epsilon,current_state,currently_available_actions):
    if np.random.random() < epsilon:
        index = np.random.randint(0, 3)
        next_action = currently_available_actions[index]
    else:
        index = np.argmax(agent.getAction(current_state))
        next_action = currently_available_actions[index]
    return next_action, index

def closestSegmentDistance(current_state):
    distance = 50000000
    for point in current_state["snake_body_pos"]:
        relative_distance = abs(point[0] - current_state["snake_head_x"]) + abs(point[1] - current_state["snake_head_y"])
        if  relative_distance < distance:
            distance = relative_distance
    return distance

def closestWallDistance(current_state,screen_size):
    distances = []
    distances.append(current_state["snake_head_x"])
    distances.append(current_state["snake_head_y"])
    distances.append(abs(screen_size-current_state["snake_head_x"]))
    distances.append(abs(screen_size-current_state["snake_head_y"]))
    distances = np.array(distances)
    return np.min(distances)

def getInput(current_state,screen_size):
    inputData = []
    # positionare relativa la snake
    relative_x = np.sign(current_state["food_x"] - current_state["snake_head_x"])
    relative_y = np.sign(current_state["food_y"] - current_state["snake_head_y"])
    food_distance = abs(current_state["food_x"] - current_state["snake_head_x"]) +\
                    abs(current_state["food_y"] - current_state["snake_head_y"])
    inputData.append(relative_x)
    inputData.append(relative_y)
    inputData.append(food_distance)
    # cea mai apropiata coliziune
    if len(current_state["snake_body_pos"]) >= 7:
        if closestSegmentDistance(current_state) < closestWallDistance(current_state,screen_size):
            inputData.append(closestSegmentDistance(current_state))
        else:
            inputData.append(closestWallDistance(current_state,screen_size))
    else:
        inputData.append(closestWallDistance(current_state,screen_size))
    return np.array(inputData)


screen_size = 150
game = Snake(screen_size, screen_size, 3, )
p = PLE(game, fps=15, display_screen=True, force_fps=True)

agent = MyAgent()

game.init()
game.clock = pygame.time.Clock()

# initialisation part:
epsilon = 0.9
batch_size = 50
actions = sorted(list(game.getActions()))  # left, right, down, up
all_possible_actions = [[2,0,3], [3,1,2], [1,2,0], [0,3,1]]  # for left, right, down, up
currently_available_actions = all_possible_actions[1]
current_state = getInput(game.getGameState(),screen_size)
training_set = []
iterations = 0
reset_flag = False

while True:
    if game.game_over() or reset_flag:
        game.init()
        available_actions = all_possible_actions[1]
        current_state = getInput(game.getGameState(),screen_size)
        reset_flag = False
		iterations = 0

    dt = game.clock.tick_busy_loop(30)
    # get the action and its reward, compute the new currently_available_actions:
    action, index = nextAction(agent,epsilon,current_state,currently_available_actions)
    reward = p.act(actions[action])
    currently_available_actions = all_possible_actions[action]

    # Next step
    game.step(dt)
    new_state = getInput(game.getGameState(),screen_size)

    if iterations > 300:
        iterations = 0
        reset_flag = True
        reward = game.rewards["loss"]
        
    if reward > 0:
        iterations = 0

    experience = [current_state, index, reward, new_state]
    training_set.append(experience)

    # send training data:
    if len(training_set) == batch_size:
        print("Train")
        agent.train_batch(training_set)
        training_set.clear()

    current_state = new_state
    iterations += 1
    pygame.display.update()
