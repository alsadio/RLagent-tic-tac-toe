# Tic-Tac-Toe Reinforcement Learning Project

## Overview
This project implements a self-playing Tic-Tac-Toe agent using various reinforcement learning techniques. The goal is to create an AI that can learn to play Tic-Tac-Toe effectively through different learning strategies.

## Implemented Algorithms
1. **Evolutionary Strategy**: An approach that evolves a population of agents over time, selecting the best performers to create new generations.
2. **NEAT (NeuroEvolution of Augmenting Topologies)**: A genetic algorithm for the generation of evolving artificial neural networks.
3. **SARSA (State-Action-Reward-State-Action)**: An on-policy TD control algorithm for estimating action-value functions.
4. **Q-Learning**: An off-policy TD control algorithm that learns the value of an action in a particular state.

## Key Features
- Implementation of multiple reinforcement learning algorithms for comparison
- Custom Tic-Tac-Toe environment for agent training
- Visualization of learning progress and agent performance
- Flexible agent classes allowing easy experimentation with hyperparameters

## Project Structure
- `Model.py`: Contains the main model classes for Evolutionary Strategy, SARSA, and Q-Learning
- `Agents.py`: Defines the agent classes including RuleBaseAgent, SarsaMLPAgent, and QLearningMLPAgent
- `Game.py`: Implements the Tic-Tac-Toe game logic 
- `Utils.py`: Utility functions for feature construction and other helper methods
- `Player.py`: Defines the Player enum, representing the two players in the game

### Player Enum
The `Player.py` file contains an `IntEnum` definition for the players:

```python
from enum import IntEnum

class Player(IntEnum):
    PLAYER1 = 1
    PLAYER2 = -1
```

This enum is used throughout the project to represent the two players, with PLAYER1 having a value of 1 and PLAYER2 having a value of -1. This representation allows for easy manipulation of the game state and simplifies win condition checks.

## License
This project was created as part of a university course assignment at Dalhousie University. It is primarily for educational purposes and demonstration of reinforcement learning concepts
