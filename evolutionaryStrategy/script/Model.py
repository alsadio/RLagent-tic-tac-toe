import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt 
from script.Game import TicTakToe
from script.Agents import RuleBaseAgent,ESAgent
from script.Player import Player

class Evolutionary_Model:
    def __init__(self,max_pop = 100, parent_percent = 0.2):
        """initilize evolutionary model and first population

        Args:
            max_pop (int, optional): maximum population at each epoch. Defaults to 1000.
            parent_percent (float, optional): the percentage of parent after selection. Defaults to 0.2.
        
        """
        self.max_pop = max_pop
        self.parent_percent = parent_percent

        self.best_agent = None
        self.best_reward = -1000
        self.population = self.initial_population()

    def play_tic_tak_toe(self, max_epoch = 100):
        """play tic tak toe evolutionary and return the best agent you have

        you should follow the persudocode in slides but you allow to some changes as long as it return the best agent you have in the model.

        1. intilize population
        2. do
        3.  fitness         <- evaluation
        3.  parent          <- selection
        4.  new_population  <- evolution
        5. return best_agent

        Returns:
            best_agent (ESagent): _description_
        """
        self.fitness_history = []
        e_trials = 15
        for epoch in range(max_epoch):
            fitness = [self.evaluation(TicTakToe(rng=0), agent, e_trials) for agent in self.population]
            parents = self.selection(fitness, self.population)
            self.population = self.evolution(parents)
            max_reward = max(fitness)
            self.fitness_history.append(max_reward)
            if max_reward > self.best_reward:
                self.best_reward = max_reward
                self.best_agent = copy.deepcopy(self.population[fitness.index(max_reward)])
        return self.best_agent
    
    def initial_population(self):
        """initilize first population

        Returns:
            population (ESAgent[]): Array of Agents
        """
        return [ESAgent() for _ in range(self.max_pop)]
    def evaluation(self,env,agent,num_trials):
        """
            evaluate the reward for each agent. feel free to have your own reward function.
        """
        total_reward = 0
        for _ in range(num_trials):
            env.reset()
            while not env.terminate: 
                move = agent.make_a_move(env.board)
                env.step(move)
                if env.terminate:
                    break
            
            if env.winner == Player.PLAYER1:
                total_reward += 1
            elif env.winner == Player.PLAYER2:
                total_reward -= 1
            else:
                total_reward += 0
        return total_reward / num_trials

    def selection(self,rewards,population):
        """
            select the best fit in the population. feel free to have your own selection.
            Make sure you select parent according to parent_percent
        """
        num_parents = int(self.parent_percent * len(population))
        selected_indices = np.argsort(rewards)[-num_parents:]
        return [copy.deepcopy(population[i]) for i in selected_indices]

    def evolution(self,parents):
        """
            evolute new population from parents. 

            be careful about how to reinforce children. You don't want your children perform same as parents and even worser than parents.

            feel free to have your own evolution. In MLP case, you would like to add some noises to weights and bias.
        """
        new_population = []
        for _ in range(self.max_pop):
            parent = np.random.choice(parents)
            child = copy.deepcopy(parent)

            child.weights += np.random.normal(0, 0.1, size=child.weights.shape)
            child.bias += np.random.normal(0, 0.1, size=child.bias.shape)
            new_population.append(child)
        return new_population
    
    def plot_performance(self, fitness_history):
        """Plot the fitness curve over generations.

        Args:
            fitness_history (list): List of fitness values for each generation.
        """
        plt.plot(fitness_history)
        plt.xlabel('Generations')
        plt.ylabel('Max Fitness')
        plt.title('Fitness Curve Over Generations')
        plt.show()