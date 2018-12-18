from ple.games.snake import Snake
from ple import PLE
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


class MyAgent:
    def __init__(self):
        self.model = keras.models.load_model("snake_model.h5")

    def train(self, train_set, target_set):
        self.model.fit(x=train_set, y=target_set, epochs=1, verbose=0)
        print(self.model.layers[2].get_weights()[1])
        self.train_counter += 1
        if self.train_counter == 200:
            print("CHECKPOINT")
            self.model.save('snake_model.h5')
            self.train_counter = 0

    def getAction(self, state):
        return self.model.predict(state.reshape(1, 2))[0]

    def train_batch(self, train_set):
        y = 0.9
        target_set = []
        for current_state, action, reward, new_state, game_over in train_set:
            if game_over:
                target = reward
            else:
                target = reward + y * np.max(agent.getAction(new_state))
            target_vector = agent.getAction(current_state)
            target_vector[action] = target
            target_set.append(target_vector)
        train_set = [i[0] for i in train_set]
        self.train(np.array(train_set), np.array(target_set))


# get action from currently_available_actions + index:
def nextAction(agent, epsilon, current_state, currently_available_actions):
    # if np.random.random() < epsilon:
    #     index = np.random.randint(0, 3)
    #     next_action = currently_available_actions[index]
    # else:
    index = np.argmax(agent.getAction(current_state))
    print(agent.getAction(current_state))
    next_action = currently_available_actions[index]
    return next_action, index


def closestSegmentDistance(current_state):
    distance = 50000000
    for point in current_state["snake_body_pos"]:
        relative_distance = abs(
            point[0] - current_state["snake_head_x"]) + abs(point[1] - current_state["snake_head_y"])
        if relative_distance < distance:
            distance = relative_distance
    return distance


def closestWallDistance(current_state, screen_size):
    distances = []
    distances.append(current_state["snake_head_x"])
    distances.append(current_state["snake_head_y"])
    distances.append(abs(screen_size - current_state["snake_head_x"]))
    distances.append(abs(screen_size - current_state["snake_head_y"]))
    distances = np.array(distances)
    return np.min(distances)


def getInput(current_state, screen_size):
    inputData = []
    # positionare relativa la snake
    relative_x = np.sign(
        current_state["food_x"] - current_state["snake_head_x"])
    relative_y = np.sign(
        current_state["food_y"] - current_state["snake_head_y"])
    food_distance = abs(current_state["food_x"] - current_state["snake_head_x"]) +\
        abs(current_state["food_y"] - current_state["snake_head_y"])
    # inputData.append((relative_x+1.0)/2)
    # inputData.append((relative_y+1.0)/2)
    inputData.append(food_distance)
    # cea mai apropiata coliziune
    # if len(current_state["snake_body_pos"]) >= 7:
    #     if closestSegmentDistance(current_state) < closestWallDistance(current_state, screen_size):
    #         inputData.append(closestSegmentDistance(current_state))
    #     else:
    #         inputData.append(closestWallDistance(current_state, screen_size))
    # else:
    inputData.append(closestWallDistance(current_state, screen_size))
    return np.array(inputData)


screen_size = 150
game = Snake(screen_size, screen_size, 3, )
p = PLE(game, fps=15, display_screen=True, force_fps=True)

agent = MyAgent()

p.init()

# initialisation part:
game.rewards["loss"] = -2
game.rewards["positive"] = 1.01
epsilon = 0.9
batch_size = 300
actions = sorted(list(game.getActions()))  # left, right, down, up
all_possible_actions = [[2, 0, 3], [3, 1, 2], [
    1, 2, 0], [0, 3, 1]]  # for left, right, down, up
currently_available_actions = all_possible_actions[1]
current_state = getInput(game.getGameState(), screen_size)
training_set = []
iterations = 0
reset_flag = False
maxscore = 0
game_over_flag = False
current_score = 0
prev_score = current_score
prev_epsilon = epsilon
while True:
    if game.game_over() is True or reset_flag:
        # if prev_score > 0:
        #     if np.random.random() < prev_score:
        #         print(epsilon)
        #         prev_epsilon = epsilon
        #         agent.train_batch(training_set)
        #     else:
        #         epsilon = prev_epsilon
        # elif np.random.random() < 0.01:
        #     print(epsilon)
        #     prev_epsilon = epsilon
        #     agent.train_batch(training_set)
        # else:
        #     epsilon = prev_epsilon
        # training_set.clear()
        p.init()
        available_actions = all_possible_actions[1]
        # current_state = getInput(game.getGameState(), screen_size)
        reset_flag = False
        iterations = 0
        game_over_flag = True

    dt = game.clock.tick_busy_loop(30)
    # get the action and its reward, compute the new currently_available_actions:
    action, index = nextAction(
        agent, epsilon, current_state, currently_available_actions)
    reward = p.act(actions[action])
    currently_available_actions = all_possible_actions[action]
    #
    # # Next step
    # # game.step(dt)
    new_state = getInput(game.getGameState(), screen_size)
    #
    # if iterations > 100:
    #     iterations = 0
    #     reset_flag = True
    #     reward = game.rewards["loss"]
    # if reward > 0:
    #     iterations = 0
    # prev_score = current_score
    # current_score = game.getScore()
    # if current_score > maxscore:
    #     maxscore = current_score
    #     print(maxscore)
    # # if current_score > prev_score:
    # #     # print(current_score)
    # #     pass
    # # else:
    # if current_state[2] < new_state[2]:
    #     reward += (-current_state[2] / screen_size) / 150
    # # print(reward)
    # experience = [current_state, index, reward, new_state, game.game_over()]
    # training_set.append(experience)
    current_state = new_state
    # iterations += 1
    # epsilon *= 0.9999
