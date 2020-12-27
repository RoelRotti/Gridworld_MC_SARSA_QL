import random

import numpy as np, math, scipy.stats, norm
import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib.colors as colors

grid = [[1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0
        [1, 1, 0, 0, 0, 0, 0, 1, 1],  # 1
        [1, 1, 1, 1, 1, 1, 0, 1, 1],  # 2
        [1, 1, 1, 1, 1, 1, 0, 1, 1],  # 3
        [1, 1, 1, 1, 1, 1, 0, 1, 1],  # 4
        [1, 1, 1, 1, 1, 1, 0, 1, 1],  # 5
        [1, 1, 1, 1, 1, -50, 1, 1, 1],  # 6
        [1, 0, 0, 0, 0, 1, 1, 1, 1],  # 7
        [1, 1, 1, 1, 1, 1, 1, 1, 50]]  # 8
        # 0, 1, 2, 3, 4, 5, 6, 7, 8
start = (4, 0)
win = (8, 8)
lose = (6, 5)


class State:
    def __init__(self):#, state_current=start):
        self.gridworld = grid
        # TODO: random start tile
        self.state_current = self.return_random_start()
        self.over = False

    def isOver(self):
        if self.state_current == win or self.state_current == lose:
            self.over = True

    def return_random_start(self):
        randomstart = 0
        while randomstart != 1:
            y = int(random.uniform(0,8))
            x = int(random.uniform(0,8))
            randomstart = grid[y][x]
        return (y, x)

    def returnReward(self, statest=None):
        if statest:
            if statest == win:
                return 50
            elif statest == lose:
                return -50
            else:
                return -1
        else:
            if self.state_current == win:
                return 50
            elif self.state_current == lose:
                return -50
            else:
                return -1

    def printBoard(self):
        ax = plt.subplot()
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 1:
                    ax.quiver(j, i, 1, 1)
                elif grid[i][j] == 0:
                    ax.quiver(j, i, 1, 1)
                elif grid[i][j] == 3:
                    ax.quiver(j, i, 1, 1)
                elif grid[i][j] == -3:
                    ax.quiver(j, i, -1, 1)
        # Reduce variance for imshow
        self.gridworld[win[0]][win[1]] = 3
        self.gridworld[lose[0]][lose[1]] = -3
        plt.title("Gridworld")
        plt.imshow(self.gridworld)
        plt.legend()
        plt.show()

    def nextState(self, action):
        # states = self.state
        # y = self.state[0]
        # x = self.state[1]
        # print(self.state_current)
        # print(self.state_current[0], self.state_current[1])
        # print(y, x)
        if action == "north":
            nextStates = (self.state_current[0] - 1, self.state_current[1])
        elif action == "south":
            nextStates = (self.state_current[0] + 1, self.state_current[1])
        elif action == "east":
            nextStates = (self.state_current[0], self.state_current[1] + 1)
        else:  # action == "west":
            nextStates = (self.state_current[0], self.state_current[1] - 1)

        # check for edges:
        if (nextStates[0] >= 0) and (nextStates[0] < 9) and (nextStates[1] >= 0) and (nextStates[1] < 9):
            # check for walls
            if self.gridworld[nextStates[0]][nextStates[1]] != 0:
                return nextStates
        return self.state_current


class Agent:
    def __init__(self):
        self.actions = ["north", "south", "east", "west"]
        self.State = State()
        self.value_function = numpy.zeros([9, 9])
        self.total_reward = numpy.zeros([9, 9])
        self.total_visits = numpy.zeros([9, 9])
        self.visit = numpy.zeros([9, 9])
        self.gamma = 0.5
        self.number_iterations = 0
        self.states = []

        #Sarsa
        self.alpha = 0
        self.e = 0.1
        self.q_values = numpy.zeros([9, 9])

    def e_greedy(self, statest = None):
        if statest:
            states = statest
        else:
            states = self.states[-1]
        if random.uniform(0,1) >= 0.1:
            best_neighbours = self.return_best_neighbours(states[0], states[1], True)
            action_index = random.choice(best_neighbours)
            if states == (8,7): action_index = 2
            if states == (7, 8): action_index = 1
            return self.actions[action_index]
        else:
            #TODO: explore non-optimal actions?
            action = random.choice(self.actions)
            return action

    def SARSA_greedy(self, episodes, gamma, alpha, e):
        self.number_iterations = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.e = e
        # Initialize Q(s,a) arbitrarily
        #self.q_values = numpy.zeros([9, 9])
        self.q_values = [[[0 for k in range(4)] for j in range(9)] for i in range(9)]
        for i in range(episodes):
            # Initialize s
            self.states.append(self.State.state_current)
            # Choose a from s using e-greedy
            action = self.e_greedy()
            # Repeat for each step of episode
            while self.State.over is False:
                # Take action a, observe r, s'
                new_state = self.takeAction(action)
                reward = self.State.returnReward(new_state)
                current_action_index = self.actions.index(action)
                # Choose a' from s' using e=-greedy
                action = self.e_greedy(new_state)
                new_action_index = self.actions.index(action)
                # Q(s,a) <- Q(s,a) + a(r + gamma*Q(s',a') - Q(s,a))
                #TODO: more recursive?
                #q_prime = self.State.returnReward(self.State.nextState(action))

                q_prime = self.q_values[new_state[0]][new_state[1]][new_action_index]
                q_s_a = self.q_values[self.states[-1][0]][self.states[-1][1]][current_action_index]
                self.q_values[self.states[-1][0]][self.states[-1][1]][current_action_index] += self.alpha * (reward + self.gamma * q_prime - q_s_a)
                    #q_s_a + \
                    #self.alpha * (reward + self.gamma * q_prime - q_s_a)
                # s<-s', a<-a'
                self.State.state_current = new_state
                self.states.append(new_state)
                self.State.isOver()
            self.reset()
            # Until s is terminal

    def Q_learning(self, episodes, gamma, alpha):
        self.number_iterations = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.q_values = numpy.zeros([9, 9])
        for i in range(episodes):
            # Initialize s
            # TODO: random start
            self.states.append(start)
            # Choose a from s using e-greedy

            # Repeat for each step of episode

            # Take action a, observe r, s'

            # Choose a' from s' using e=-greedy

            # Q(s,a) <- Q(s,a) + a(r + gamma*Q(s',a') - Q(s,a))

            # s<-s', a<-a'
            # Until s is terminal


    def equiprobableAction(self):
        action = random.choice(self.actions)
        return action

    def takeAction(self, action):
        nextStates = self.State.nextState(action)
        #return State(state_current=nextStates)
        return nextStates

    def MonteCarlo(self, iterations, gamma):
        self.number_iterations = iterations
        self.gamma = gamma
        for i in range(iterations):
            self.MCpath()
            self.backUpMC()
            self.reset()

    def MCpath(self):
        while self.State.over is False:
            # Pick action
            action = self.equiprobableAction()
            # Go to next state
            self.State.state_current = self.takeAction(action)
            # Add to the list of states
            self.states.append(self.State.state_current)
            # Check if end tile has been reached
            self.State.isOver()

    def backUpMC(self):
        first = True
        G = 0
        self.visit = numpy.zeros([9, 9])
        #for i in reversed(range(len(self.states[0:-1]))):  # starts at final last of list, and descends to first state
        for i in reversed(range(len(self.states))):  # starts at final last of list, and descends to first state
            # Coordinates:
            y = self.states[i][0]
            x = self.states[i][1]
            # G calculation
            if first:
                G = self.State.returnReward(self.states[i])
                first = False
            else:
                G = self.gamma * G + self.State.returnReward(self.states[i + 1])
            if self.visit[y][x] == 0:
                self.total_reward[y][x] += G
                # number of visits:
                self.total_visits[y][x] += 1
                # visits this episode:
                self.visit[y][x] = 1
                # Update value function:
                self.value_function[y][x] = self.total_reward[y][x] / self.total_visits[y][x]

    def show_value_function(self, SARSA=False):
        ax = plt.subplot()
        for i in range(9):
            for j in range(9):
                if grid[i][j] != 0 and grid[i][j] != 50 and grid[i][j] != -50:
                    if SARSA:
                        best_neighbour = [self.q_values[i][j].index(max(self.q_values[i][j]))]
                    else:
                        best_neighbour = self.return_best_neighbours(i, j)
                    if 0 in best_neighbour:# == "north":
                        ax.quiver(j, i, 0, 1)
                    if 1 in best_neighbour:# == "south":
                        ax.quiver(j, i, 0, -1)
                    if 2 in best_neighbour:# == "east":
                        ax.quiver(j, i, 1, 0)
                    if 3 in best_neighbour:# == "west":
                        ax.quiver(j, i, -1, 0)
        if SARSA:
            plt.title(
                "Q-value function gridworld. Gamma = {}. # iterations = {}\n Alpha = {}. e = {}".format(self.gamma, self.number_iterations, self.alpha, self.e))
            for i in range (9):
                print (i , self.q_values[i], i)
            self.q_values = [[max(self.q_values[i][j]) for j in range(9)] for i in range(9)]
            plt.imshow(self.q_values)
        else:
            plt.title("Value function gridworld after MC policy evaluation for\n equiprobable policy. Gamma = {}. # iterations = {}\n Absolute tolerance arrows = 0.0001".format(self.gamma, self.number_iterations))
            plt.imshow(self.value_function)
            plt.clim(-5, 5)
        plt.subplots_adjust(top=0.85)
        plt.colorbar()
        plt.show()

    def return_best_neighbours(self, y, x, SARSA=False):
        neighbours = [-math.inf, -math.inf, -math.inf, -math.inf]
        #north
        if y-1 >= 0:
            if grid[y-1][x] != 0:
                if SARSA: neighbours[0] = max(self.q_values[y-1][x])
                else: neighbours[0] = self.value_function[y-1][x]
        #south
        if y+1 < 9:
            if grid[y+1][x] != 0:
                if SARSA: neighbours[1] = max(self.q_values[y+1][x])
                else: neighbours[1] = self.value_function[y+1][x]
        #east
        if x+1 < 9:
            if grid[y][x+1] != 0:
                if SARSA: neighbours[2] = max(self.q_values[y][x+1])
                else: neighbours[2] = self.value_function[y][x+1]
        #west
        if x-1 >= 0:
            if grid[y][x-1] != 0:
                if SARSA: neighbours[3] = max(self.q_values[y][x-1])
                else: neighbours[3] = self.value_function[y][x-1]
        max_value = max(neighbours)
        indices = [index for index, value in enumerate(neighbours) if math.isclose(value, max_value, abs_tol=0.0001)]#value == max_value]
        return indices

    def reset(self):
        self.states = []
        self.State = State()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    agent = Agent()
    #agent.MonteCarlo(10000)
    agent.SARSA_greedy(1000, 0.8, 0.5, 0.1)
    agent.show_value_function(SARSA=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
