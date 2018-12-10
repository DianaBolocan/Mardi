from ple.games.snake import Snake
from ple import PLE
import pygame
import keras
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout

class MyAgent():
    def __init__(self):
        model = keras.models.Sequential()
        model.add(InputLayer((6,)))  # head.x/size, head.y/size, fruit.x/size, fruit.y/size, min(d(head,body)), no.segments
        model.add(Dense(100, activation = "sigmoid"))
        model.add(Dense(4, activation = "softmax"))

game = Snake(150,150,3)
p = PLE(game, fps=30, display_screen=True)
#agent = myAgentHere(allowed_actions=p.getActionSet())

game.init()
reward = 0.0

game.clock = pygame.time.Clock()
curr_dist=0
print(p.getActionSet())

# actios board directions X0Y: {
#     119: up
#     97: left
#     100: right
#     115: down
# }

while True:
    if game.game_over():
        game.init()
    # print(p.score())
    dt = game.clock.tick_busy_loop(30)
    game.step(dt)
    print(p.act(119))
    p.act(97)
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