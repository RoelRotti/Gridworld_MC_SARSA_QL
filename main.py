import random

import numpy as np, math, scipy.stats, norm
import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import make_interp_spline, BSpline

# Define the gridworld: map + win/lose
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

# Class that represents the position within the gridworld. Mainly for checking valid moves
class State:
    def __init__(self):#, state_current=start):
        self.gridworld = grid
        self.state_current = self.return_random_start()
        self.over = False
        self.statesState = []

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

    def printBoard(self, doPrint=True, path=True):
        world = self.gridworld
        if doPrint:
            if path:
                for s in range(len(self.statesState)):
                    world[self.statesState[s][0]][self.statesState[s][1]] = 'A'
            else:
                world[self.state_current[0]][self.state_current[1]] = 'A'
            print(np.matrix(world))
        else:
            # Reduce variance for imshow to show difference in colour
            world[win[0]][win[1]] = 2
            world[lose[0]][lose[1]] = -2
            # Set current state
            if path:
                for s in range(len(self.statesState)):
                    world[self.statesState[s][0]][self.statesState[s][1]] = 3
            else:
                world[self.state_current[0]][self.state_current[1]] = 3
            plt.title("Gridworld")
            plt.imshow(world)
            plt.show()

    def nextState(self, action):
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
                self.statesState.append(nextStates)
                return nextStates
        return self.state_current


## Class that represents the agent traversing through the gridworld. The 3 main algorithms that are implemented:
# -Monte Carlo Policy Evaluation
# -SARSA
# -Q-learning
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
        self.total_reward_episodes = []

        #SARSA/QL
        self.alpha = 0
        self.e = 0.1
        self.q_values = [[[0 for k in range(4)] for j in range(9)] for i in range(9)]
        self.episode = 0

    # e-greedy function that first checks for rewards and later for q-values. 'e' descends over time to 0 at the end of
    # the total amounts of episodes. Can be used with a given state or if not defined just checks the last state
    def e_greedy(self, statest=None):
        if statest:
            states = statest
        else:
            states = self.states[-1]
        # Decreasing epsilon to reduce exploration over time
        if random.uniform(0, 1) >= (self.e-self.e*(self.episode/self.number_iterations)):
            # First check for the reward, since q value terminal state = 0
            terminal, best_neighbours = self.return_best_neighbours(states[0], states[1], reward=True)
            action_index = random.choice(best_neighbours)
            if terminal == "terminal":
                return self.actions[action_index]
            else:
                terminal, best_neighbours = self.return_best_neighbours(states[0], states[1], SARSA=True)
                action_index = random.choice(best_neighbours)
            return self.actions[action_index]
        else:
            action = random.choice(self.actions)
            return action

    # SARSA-greedy based on pseudoalgorithm by Sutton & Barto
    def SARSA_greedy(self, episodes, gamma, alpha, e):
        self.number_iterations = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.e = e
        for i in range(episodes):
            self.episode = i
            # Initialize s
            self.states.append(self.State.state_current)
            self.total_reward_episodes.append(0)# Keep track of reward per episode
            # Choose a from s using e-greedy
            action = self.e_greedy()
            # Repeat for each step of episode
            while self.State.over is False:
                # Take action a, observe r, s'
                new_state = self.takeAction(action)
                reward = self.State.returnReward(new_state)
                self.total_reward_episodes[-1] += reward # Keep track of reward per episode
                current_action_index = self.actions.index(action)
                # Choose a' from s' using e=-greedy
                action = self.e_greedy(new_state)
                new_action_index = self.actions.index(action)
                # Q(s,a) <- Q(s,a) + a(r + gamma*Q(s',a') - Q(s,a))
                q_prime = self.q_values[new_state[0]][new_state[1]][new_action_index]
                q_s_a = self.q_values[self.states[-1][0]][self.states[-1][1]][current_action_index]
                self.q_values[self.states[-1][0]][self.states[-1][1]][current_action_index] += self.alpha * (reward + self.gamma * q_prime - q_s_a)
                    #q_s_a + \
                    #self.alpha * (reward + self.gamma * q_prime - q_s_a)
                # s<-s', a<-a'
                self.State.state_current = new_state
                self.states.append(new_state)
                self.State.isOver()
            #self.State.printBoard(doPrint=False)
            self.reset()
            # Until s is terminal

    # Q-Learning based on pseudoalgorithm by Sutton & Barto
    def Q_learning(self, episodes, gamma, alpha, e):
        self.number_iterations = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.e = e
        for i in range(episodes):
            # Initialize s
            self.states.append(self.State.state_current)
            self.total_reward_episodes.append(0)  # Keep track of reward per episode
            self.episode = i
            # Repeat for each step of episode
            while self.State.over is False:
                # Choose a from s using policy from Q (e.g e-greedy)
                action = self.e_greedy()
                # Take action a, observe r, s'
                new_state = self.takeAction(action)
                reward = self.State.returnReward(new_state)
                self.total_reward_episodes[-1] += reward  # Keep track of reward per episode
                # Q(s,a) <- Q(s,a) + a(r + gamma* max(a)Q(s',a) - Q(s,a))
                current_action_index = self.actions.index(action)
                maxa_qsa = max(self.q_values[new_state[0]][new_state[1]])
                self.q_values[self.states[-1][0]][self.states[-1][1]][current_action_index] += self.alpha * (
                            reward + self.gamma * maxa_qsa - self.q_values[self.states[-1][0]][self.states[-1][1]][current_action_index])
                # s<-s',
                self.states.append(new_state)
                self.State.state_current = new_state
                self.State.isOver()
            # Until s is terminal
            self.reset()

    # Select a random policy out of the actions of the agent
    def equiprobableAction(self):
        action = random.choice(self.actions)
        return action

    # Agents takes action. In State it is checked whether this is a valid action
    def takeAction(self, action):
        nextStates = self.State.nextState(action)
        return nextStates

    # Monte Carlo Policy Evaluation based on pseudoalgorithm by Sutton & Barto
    def MonteCarlo(self, iterations, gamma):
        self.number_iterations = iterations
        self.gamma = gamma
        for i in range(iterations):
            self.MCpath()
            reward = self.backUpMC()
            self.total_reward_episodes.append(reward)
            self.reset()

    # Helper function MCPE. Completes one episode by taking a path to terminal state
    def MCpath(self):
        while self.State.over is False:
            # Pick action
            action = self.equiprobableAction()
            # Go to next state
            self.State.state_current = self.takeAction(action)
            # Add to the list of states
            self.states.append(self.State.state_current)
            self.State.statesState.append(self.State.state_current)
            # Check if end tile has been reached
            self.State.isOver()

    # Helper function MCPE. Backs up path after 'MCpath' and assigns correct v-values
    def backUpMC(self):
        first = True
        G = 0
        self.visit = numpy.zeros([9, 9])
        reward_path = 0
        ## For actual implementation: use lower for loop. Terminal tiles will be 0 then.
        for i in reversed(range(len(self.states[0:-1]))):  # starts at final last of list, and descends to first state
        for i in reversed(range(len(self.states))):  # starts at last of list, and descends to first state
            # Coordinates:
            y = self.states[i][0]
            x = self.states[i][1]
            # G calculation
            ## For actual implementation: use only else: Terminal tiles will be 0 then.
            if first:
                G = self.State.returnReward(self.states[i])
                first = False
            else:
                G = self.gamma * G + self.State.returnReward(self.states[i + 1])
            if self.visit[y][x] == 0:
                ## When actual implementation: change i
                reward_path += self.State.returnReward(self.states[i])
                self.total_reward[y][x] += G
                # number of visits:
                self.total_visits[y][x] += 1
                # visits this episode:
                self.visit[y][x] = 1
                # Update value function:
                self.value_function[y][x] = self.total_reward[y][x] / self.total_visits[y][x]
        # For keeping track of reward over time:
        return reward_path

    # Helper function to plot the heat map of the gridworld. Can be used for MC, SARSA, QL
    def show_value_function(self, SARSA=False, QL=False):
        ax = plt.subplot()
        for i in range(9):
            for j in range(9):
                if grid[i][j] != 0 and grid[i][j] != 50 and grid[i][j] != -50:
                    if SARSA or QL:
                        best_neighbour = [self.q_values[i][j].index(max(self.q_values[i][j]))]
                    else:
                        terminal, best_neighbour = self.return_best_neighbours(i, j)
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
                "SARSA: Optimal policy in gridworld. Gamma = {} \n # iterations = {} Alpha = {}. e = {}".format(self.gamma,
                                                                                                               self.number_iterations,
                                                                                                               self.alpha, self.e))
            # for i in range (9):
            #     print (i , self.q_values[i], i)
            self.q_values = [[max(self.q_values[i][j]) for j in range(9)] for i in range(9)]
            plt.imshow(self.q_values)
        elif QL:
            plt.title(
                "Q-Learning: Optimal policy in gridworld. Gamma = {} \n # iterations = {} Alpha = {}. e = {}".format(self.gamma,
                                                                                                        self.number_iterations,
                                                                                                        self.alpha,
                                                                                                        self.e))
            # for i in range(9):
            #     print(i, self.q_values[i], i)
            self.q_values = [[max(self.q_values[i][j]) for j in range(9)] for i in range(9)]
            plt.imshow(self.q_values)
        else:
            plt.title("Value function gridworld after MC policy evaluation for\n equiprobable policy. Gamma = {}. # iterations = {}\n Absolute tolerance arrows = 0.0001".format(self.gamma, self.number_iterations))
            plt.imshow(self.value_function)
            #plt.clim(-5, 5)
        plt.subplots_adjust(top=0.85)
        plt.colorbar()
        plt.show()

    # Helper function. Is used in various ways throughout the code such as for printing, SARSA, checking reward etc.
    def return_best_neighbours(self, y, x, SARSA=False, reward=False):
        neighbours = [-math.inf, -math.inf, -math.inf, -math.inf]
        #north
        if y-1 >= 0:
            if grid[y-1][x] != 0:
                if SARSA: neighbours[0] = max(self.q_values[y-1][x])
                elif reward: neighbours[0] = self.State.returnReward((y-1, x))
                else: neighbours[0] = self.value_function[y-1][x]
        #south
        if y+1 < 9:
            if grid[y+1][x] != 0:
                if SARSA: neighbours[1] = max(self.q_values[y+1][x])
                elif reward: neighbours[1] = self.State.returnReward((y + 1, x))
                else: neighbours[1] = self.value_function[y+1][x]
        #east
        if x+1 < 9:
            if grid[y][x+1] != 0:
                if SARSA: neighbours[2] = max(self.q_values[y][x+1])
                elif reward: neighbours[2] = self.State.returnReward((y, x+1))
                else: neighbours[2] = self.value_function[y][x+1]
        #west
        if x-1 >= 0:
            if grid[y][x-1] != 0:
                if SARSA: neighbours[3] = max(self.q_values[y][x-1])
                elif reward: neighbours[3] = self.State.returnReward((y, x-1))
                else: neighbours[3] = self.value_function[y][x-1]
        max_value = max(neighbours)
        indices = [index for index, value in enumerate(neighbours) if math.isclose(value, max_value, abs_tol=0.0001)]#value == max_value]
        # For reward: since q value terminal node is 0 also check for reward
        if reward and max_value == 50:
            return "terminal", indices
        return None, indices

    # Correct resets for after one episode
    def reset(self):
        self.states = []
        self.State = State()


# Printer function to print return reward over episodes for the equiprobable policy, SARSA & QL
def plot_total_reward(reward1, reward2, reward3):
    x = []
    for i in range(len(reward1)):
        x.append(i)
    plt.figure()
    poly = np.polyfit(x, reward1, 10)
    poly_y = np.poly1d(poly)(x)
    poly2 = np.polyfit(x, reward2, 10)
    poly_y2 = np.poly1d(poly2)(x)
    poly3 = np.polyfit(x, reward3, 10)
    poly_y3 = np.poly1d(poly3)(x)
    plt.plot(x, poly_y, label="Reward MCPE equiprobable policy")
    plt.plot(x, poly_y2, label="Reward SARSA")
    plt.plot(x, poly_y3, label="Reward Q-Learning")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Reward per episode for MCPE equiprobable policy & SARSA & Q-learning\n Gamma = {}, alpha = {}, e = {}".format(0.8, 0.5, 0.1))
    plt.legend()
    plt.show()


# Main: set parameters below to run different options
if __name__ == '__main__':
    v_valueMC = 0
    q_valueSARSA = 0
    q_valueQL = 1
    reward_over_episodes = 0

    # Plot v-values MC
    if v_valueMC:
        agent = Agent()
        agent.MonteCarlo(10000, 0.5)

    # Plot q values SARSA/QL:
    if q_valueSARSA:
        agent = Agent()
        agent.SARSA_greedy(1000, 0.8, 0.5, 0.1)
        agent.show_value_function(SARSA=True)

    # Plot q values SARSA/QL:
    if q_valueQL:
        agent = Agent()
        agent.Q_learning(100000, 0.8, 0.5, 0.1)
        agent.show_value_function(QL=True)

    # Plot reward Equiprobable Policy & SARSA & QL:
    if reward_over_episodes:
        agent1 = Agent()
        agent2 = Agent()
        agent3 = Agent()
        agent1.MonteCarlo(1000, 0.8)
        agent2.SARSA_greedy(1000, 0.8, 0.5, 0.1)
        agent3.Q_learning(1000, 0.8, 0.5, 0.1)
        plot_total_reward(agent1.total_reward_episodes, agent2.total_reward_episodes, agent3.total_reward_episodes)
