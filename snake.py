from ple.games.snake import Snake
from ple import PLE
import pygame
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense


class MyAgent:
    def __init__(self):
        self.model = Sequential()
        # head.x/size, head.y/size, fruit.x/size, fruit.y/size,
        # min(d(head,body)), no.segments
        self.model.add(InputLayer((6,)))
        self.model.add(Dense(18, activation="sigmoid"))
        self.model.add(Dense(3, activation="softmax"))
        # opt = SGD(lr=0.01)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer='adam', metrics=["accuracy"])

    def train(self, train_set, target_set):
        self.model.fit(train_set.reshape(1, 6),
                       target_set.reshape(1, 3), verbose=0)

    def getAction(self, state):
        # print(state)
        # print(self.model.layers[0].get_weights())
        return self.model.predict(state.reshape(1, 6))


def getDistance(state):
    distance = abs(state[0] - state[2]) + abs(state[1] - state[3])
    return distance


def getInput(state, screenDimension):
    inputData = []
    inputData.append(state["snake_head_x"] / screenDimension)
    inputData.append(state["snake_head_y"] / screenDimension)
    inputData.append(state["food_x"] / screenDimension)
    inputData.append(state["food_y"] / screenDimension)
    inputData.append(min(state["snake_body"][2:]))
    angle = 0
    if state["snake_head_x"] - state["food_x"] > 0:
        angle += 1
    if state["snake_head_y"] - state["food_y"] > 0:
        angle += 2
    # inputData.append(len(state["snake_body"]))
    inputData.append(angle)
    return np.array(inputData)


screenDims = 150
game = Snake(screenDims, screenDims, 3)
p = PLE(game, fps=30, display_screen=True)

agent = MyAgent()

game.init()
# reward = 0.0

game.clock = pygame.time.Clock()
# curr_dist = 0
state = getInput(game.getGameState(), screenDims)
# actios board directions X0Y: {
#     119: up
#     97: left
#     100: right
#     115: down
# }
# snake direction:
# 0: right
# 1: left
# 2: down
# 3: up

# 302-right
# 213-left
# 021-down
# 130-up
directions = [[3, 0, 2], [2, 1, 3], [0, 2, 1], [1, 3, 0]]

''' NN parameters '''
game.rewards["tick"] = -1.0
game.rewards["loss"] = -150.0
game.rewards["win"] = 150.0
y = 0.9
epsilon = 0.6
iteration = 0
train_set = []
actions = list(game.getActions())
# turn left, forward, turn right
available_actions = directions[0]

while True:
    if game.game_over():
        game.init()
    dt = game.clock.tick_busy_loop(30)
    if np.random.random() < epsilon:
        next_action = np.random.randint(0, 3)
    else:

        next_action = np.argmax(agent.getAction(state))
    # Act
    # pare ciudat? dar merge
    reward = p.act(actions[available_actions[next_action]])
    # Next step
    # pare ciudat dar merge
    available_actions = directions[available_actions[next_action]]
    game.step(dt)
    new_state = getInput(game.getGameState(), screenDims)
    if getDistance(state) > getDistance(new_state):
        reward += 1
    target = reward + y * np.max(agent.getAction(new_state))
    target_set = agent.getAction(state)[0]
    target_set[next_action] = target
    agent.train(state, target_set)
    state = new_state
    iteration += 1
    epsilon *= 0.999
    # train_set.append(getInput(game.getGameState(),screenDims))
    # if iteration==500:
    #     agent.train(train_set)
    #     train_set.clear()
    pygame.display.update()

# OUR JOB LOOKS LIKE THIS
# --------------------------------------------------------------
# game = FlappyBird()
# p = PLE(game, fps=30, display_screen=True)
# agent = myAgentHere(allowed_actions=p.getActionSet())  <-- RN
#
# p.init()
# reward = 0.0
#
# for i in range(nb_frames):
#     if p.game_over():
#         p.reset_game()
#
#     observation = p.getScreenRGB()
#     action = agent.pickAction(reward, observation)
#     reward = p.act(action)


# import pygame
# import numpy as np
#
# pygame.init()
# game = Snake(width=128, height=128)
# game.screen = pygame.display.set_mode( game.getScreenDims(), 0, 32)
# game.clock = pygame.time.Clock()
# game.rng = np.random.RandomState(24)
# game.init()
#
# while True:
#     if game.game_over():
#         game.init()
#
#     dt = game.clock.tick_busy_loop(30)
#     game.step(dt)
#     pygame.display.update()
#     print(game.getGameState())
