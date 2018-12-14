from ple.games.snake import Snake
from ple import PLE
import pygame
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


class MyAgent:
    def __init__(self):
        self.model = Sequential()
        # head.x/size, head.y/size, fruit.x/size, fruit.y/size,
        # min(d(head,body)), no.segments
        self.model.add(Dense(9, input_dim=5, activation="relu"))
        self.model.add(Dense(36, activation="relu"))
        self.model.add(Dense(3))
        # opt = SGD(lr=0.01)
        # self.model.compile(loss="categorical_crossentropy",
        #                    optimizer='adam', metrics=["accuracy"])
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(sgd, "mse")

    def train(self, train_set, target_set):
        self.model.fit(x=train_set,
                       y=target_set, epochs=100, verbose=0)
        print(self.model.layers[2].get_weights())

    def getAction(self, state):
        # print(state)
        # print(self.model.layers[0].get_weights())
        # print(self.model.get_weights())
        return self.model.predict(state.reshape(1, 5))
    def train_batch(self, train_set):
        y = 0.95
        target_set = []
        for state, new_state, next_action, reward, _ in train_set:
            target = reward + y * np.max(agent.getAction(new_state))
            target_vector = agent.getAction(state)[0]
            target_vector[next_action] = target
            target_set.append(target_vector)
        train_set = [i[0] for i in train_set]
        # final_set = []
        # for i in range(len(train_set)):
        #     final_set.append([train_set[i], target_set[i]])
        # final_set = np.array(final_set)
        # np.random.shuffle(final_set)
        # train_set = np.array([i[0] for i in final_set])
        # target_set = np.array([i[1] for i in final_set])
        self.train(np.array(train_set), np.array(target_set))


def getDistance(state):
    distance = abs(state[0] - state[2]) + abs(state[1] - state[3])
    # distance = np.sqrt((state[0] - state[2])**2 + (state[1] - state[3])**2)
    return distance


def getInput(state, screenDimension, direction):
    inputData = []
    wall_ahead = 0
    inputData.append(state["snake_head_x"] / screenDimension)
    inputData.append(state["snake_head_y"] / screenDimension)
    inputData.append(state["food_x"] / screenDimension)
    inputData.append(state["food_y"] / screenDimension)
    # inputData.append(min(state["snake_body"][2:]))
    angle = np.arctan2(inputData[0] - inputData[2],
                       inputData[1] - inputData[3])
    # 0.995
    if inputData[0] > 0.995 or inputData[0] < 0.005:
        wall_ahead = 1
    if inputData[1] > 0.995 or inputData[1] < 0.005:
        wall_ahead = 1
    # inputData.append(len(state["snake_body"]))
    # inputData.append(angle)
    inputData.append(wall_ahead)
    return np.array(inputData)


# sx>fx => food to the left s
# sy>fy => food bellow      s

screenDims = 150
game = Snake(screenDims, screenDims, 3, )
p = PLE(game, fps=15, display_screen=True, force_fps=True)

agent = MyAgent()

game.init()
# reward = 0.0

game.clock = pygame.time.Clock()
# curr_dist = 0
state = getInput(game.getGameState(), screenDims, 0)
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
print(game.rewards)
directions = [[3, 0, 2], [2, 1, 3], [0, 2, 1], [1, 3, 0]]
#  return sum(self._oneStepAct(action) for i in range(self.frame_skip))
''' NN parameters '''
# game.rewards["tick"] = -1.0
# game.rewards["loss"] = -500.0
# game.rewards["tick"] = -1
# game.rewards["positive"] = 11.5
epsilon = 1
train_batch_size = 200
iteration = 0
train_iter = 0
training_set = list()
# train_set = []
actions = sorted(list(game.getActions()))
# turn left, forward, turn right
available_actions = directions[0]
flag_game_over = False
game_over = 0
reset_flag = False
maxscore = 0
prevscore = 0
while True:
    if game.game_over() or reset_flag:
        game.init()
        available_actions = directions[0]
        state = getInput(game.getGameState(), screenDims, 0)
        flag_game_over = True
        reset_flag = False
        iteration = 0
        game_over = -1
    dt = game.clock.tick_busy_loop(30)
    if np.random.random() < epsilon:
        next_action = np.random.randint(0, 3)
    else:
        next_action = np.argmax(agent.getAction(state))
    # Act
    # pare ciudat? dar merge
    # print(actions[available_actions[0]])
    reward = p.act(actions[available_actions[next_action]])
    # if flag_game_over:
    #     reward = 0.0
    #     flag_game_over = False
    print(reward)
    # Next step
    # pare ciudat dar merge
    available_actions = directions[available_actions[next_action]]
    game.step(dt)
    new_state = getInput(game.getGameState(), screenDims,
                         available_actions[next_action])
    # if getDistance(state) > getDistance(new_state):
    #     reward += 1
    # else:
    #     reward -= 1
    # print(reward)
    # print(epsilon)
    iteration += 1
    if iteration > train_batch_size // 4:
        reset_flag = True
        reward = game.rewards["loss"]
    currentscore = game.getScore()
    if prevscore < currentscore:
        iteration -= 50
    if currentscore > maxscore:
        maxscore = currentscore
        print(maxscore)
    prevscore = currentscore
    training_set.append([state, new_state, next_action, reward, game_over])
    game_over = 0
    if len(training_set) == train_batch_size:
        print("Train")
        agent.train_batch(training_set)
        training_set.clear()
    # agent.train(state, target_set)
    state = new_state
    # if np.random.random() < 0.5:
    #     iteration += 1
    #     train_set.append(state)

    # if iteration == 100:
    #     train_set.clear()
    #     iteration = 0
    epsilon *= 0.994
    # print(epsilon)
    # print(epsilon)
    # train_set.append(getInput(game.getGameState(),screenDims))
    # if iteration==500:
    #     agent.train(train_set)
    #     train_set.clear()
    # pygame.display.update()

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
