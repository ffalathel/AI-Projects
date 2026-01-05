# multiAgents.py
# --------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Addendum:
# This code was modified by Gene Kim at University of South Florida in Fall 2025
# to make solutions not match the original UC Berkeley solutions exactly and
# align with CAI 4002 course goals regarding AI tool use in projects.

from util import manhattan_distance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

util.VALIDATION_LISTS['search'] = [
    "වැසි",
    " ukupnog",
    " ᓯᒪᔪ",
    " ਪ੍ਰਕਾਸ਼",
    " podmienok",
    " sėkmingai",
    "рацыі",
    " යාපාරයා",
    "න්ද්"
]

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide. You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state: GameState):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghost_state.scared_timer for ghost_state in new_ghost_states]

        score = successor_game_state.get_score()
        
        # Get food list
        food_list = new_food.as_list()
        
        # Strong incentive to move toward closest food
        if food_list:
            food_distances = [manhattan_distance(new_pos, food) for food in food_list]
            min_food_dist = min(food_distances)
            # Use reciprocal with higher weight
            score += 1.0 / (min_food_dist + 0.1)
        
        # Evaluate ghost threat
        for ghost_state in new_ghost_states:
            ghost_pos = ghost_state.get_position()
            ghost_dist = manhattan_distance(new_pos, ghost_pos)
            
            if ghost_state.scared_timer > 0:
                # Chase scared ghosts aggressively
                score += 200.0 / (ghost_dist + 1)
            else:
                # Only worry about very close ghosts
                if ghost_dist <= 1:
                    score -= 1000
                elif ghost_dist == 2:
                    score -= 50
        
        # Strong penalty for stopping
        if action == Directions.STOP:
            score -= 100
        
        return score
 

def score_evaluation_function(current_game_state: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers. Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents. Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated. It's
    only partially specified, and designed to be extended. Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn = 'score_evaluation_function', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state: GameState):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        def minimax(state, depth, agent_index):
            #check conditions
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)
            
            #legal actions for current agent
            legal_actions = state.get_legal_actions(agent_index)
            
            #calculate next agent and depth
            next_agent = (agent_index + 1) % state.num_agents()
            next_depth = depth - 1 if next_agent == 0 else depth
            
            if agent_index == 0: #maximizing
                max_value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = minimax(successor, next_depth, next_agent)
                    max_value = max(max_value, value)
                return max_value
            else: # Ghost (minimizing)
                min_value = float('inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = minimax(successor, next_depth, next_agent)
                    min_value = min(min_value, value)
                return min_value
        
        #best action for Pacman
        legal_actions = game_state.get_legal_actions(0)
        best_action = None
        best_value = float('-inf')
        
        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = minimax(successor, self.depth, 1)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        def alpha_beta(state, depth, agent_index, alpha, beta):
            # Check terminal conditions
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)
            
            #legal actions for current agent
            legal_actions = state.get_legal_actions(agent_index)
            
            #calculate next agent and depth
            next_agent = (agent_index + 1) % state.num_agents()
            next_depth = depth - 1 if next_agent == 0 else depth
            
            if agent_index == 0: #maximizing
                value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alpha_beta(successor, next_depth, next_agent, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else: # Ghost (minimizing)
                value = float('inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, alpha_beta(successor, next_depth, next_agent, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        
        #best action for Pacman
        legal_actions = game_state.get_legal_actions(0)
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = alpha_beta(successor, self.depth, 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)
        
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agent_index):
            #terminal conditions
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)
            
            #legal actions for current agent
            legal_actions = state.get_legal_actions(agent_index)
            
            #calculate next agent and depth
            next_agent = (agent_index + 1) % state.num_agents()
            next_depth = depth - 1 if next_agent == 0 else depth
            
            if agent_index == 0: #maximizing
                max_value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = expectimax(successor, next_depth, next_agent)
                    max_value = max(max_value, value)
                return max_value
            else: #expectation over random actions
                total_value = 0
                probability = 1.0 / len(legal_actions)
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = expectimax(successor, next_depth, next_agent)
                    total_value += probability * value
                return total_value
        
        #best action for Pacman
        legal_actions = game_state.get_legal_actions(0)
        best_action = None
        best_value = float('-inf')
        
        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = expectimax(successor, self.depth, 1)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action

def better_evaluation_function(current_game_state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    #base score
    score = current_game_state.get_score()
    
    #pacman position
    pac_pos = current_game_state.get_pacman_position()
    
    #food information
    food_grid = current_game_state.get_food()
    food_list = food_grid.as_list()
    num_food = len(food_list)
    
    #ghost information
    ghost_states = current_game_state.get_ghost_states()
    
    #capsule information
    capsules = current_game_state.get_capsules()
    num_capsules = len(capsules)
    
    #terminal states
    if current_game_state.is_win():
        return 999999
    if current_game_state.is_lose():
        return -999999
    
    #prioritize closest food and penalize remaining food
    if food_list:
        min_food_dist = min([manhattan_distance(pac_pos, food) for food in food_list])
        score += 10.0 / (min_food_dist + 1)
    score -= 4 * num_food
    
    #capsule score
    if capsules:
        min_capsule_dist = min([manhattan_distance(pac_pos, capsule) for capsule in capsules])
        score += 20.0 / (min_capsule_dist + 1)
    score -= 10 * num_capsules
    
    #ghost evaluation
    for ghost in ghost_states:
        ghost_pos = ghost.get_position()
        ghost_dist = manhattan_distance(pac_pos, ghost_pos)
        
        if ghost.scared_timer > 0:
            # Chase scared ghosts
            if ghost_dist == 0:
                score += 200
            else:
                score += 100.0 / ghost_dist
        else:
            # Avoid active ghosts
            if ghost_dist < 2:
                score -= 1000
            elif ghost_dist < 4:
                score -= 200.0 / (ghost_dist + 1)
    
    return score

# Abbreviation
better = better_evaluation_function