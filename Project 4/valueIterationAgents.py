# valueIterationAgents.py
# -----------------------
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

# valueIterationAgents.py
# -----------------------
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

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

util.VALIDATION_LISTS['reinforcement'] = [
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

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state, action, next_state)
              mdp.is_terminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.run_value_iteration()

    def run_value_iteration(self):
        # batch value iteration - update all states at once
        for iteration in range(self.iterations):
            new_values = util.Counter()
            
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    new_values[state] = 0.0
                else:
                    possible_actions = self.mdp.get_possible_actions(state)
                    if not possible_actions:
                        new_values[state] = 0.0
                    else:
                        # find best action's q-value
                        max_value = float('-inf')
                        for action in possible_actions:
                            q_value = self.compute_q_value_from_values(state, action)
                            if q_value > max_value:
                                max_value = q_value
                        new_values[state] = max_value
            
            # update all values at once (batch style)
            self.values = new_values

    def get_value(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # sum over all possible next states
        q_value = 0.0
        transitions = self.mdp.get_transition_states_and_probs(state, action)
        for next_state, prob in transitions:
            reward = self.mdp.get_reward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit. Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.is_terminal(state):
            return None
        
        possible_actions = self.mdp.get_possible_actions(state)
        if not possible_actions:
            return None
        
        # pick action with highest q-value
        best_action = None
        best_q_value = float('-inf')
        
        for action in possible_actions:
            q_value = self.compute_q_value_from_values(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action

    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        "Returns the policy at the state (no exploration)."
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        return self.compute_q_value_from_values(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state)
              mdp.is_terminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """
        Run cyclic asynchronous value iteration.
        In each iteration, update only one state in a cyclic manner.
        If the state is terminal, skip it (do nothing in that iteration).
        """
        states = self.mdp.get_states()
        
        for iteration in range(self.iterations):
            # cycle through states one at a time
            state_index = iteration % len(states)
            state = states[state_index]
            
            if self.mdp.is_terminal(state):
                continue
            
            # update this state in-place
            possible_actions = self.mdp.get_possible_actions(state)
            if not possible_actions:
                self.values[state] = 0.0
            else:
                # find best q-value and update
                max_value = float('-inf')
                for action in possible_actions:
                    q_value = self.compute_q_value_from_values(state, action)
                    if q_value > max_value:
                        max_value = q_value
                self.values[state] = max_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """
        Run prioritized sweeping value iteration.
        Updates states based on priority (error magnitude), focusing on states
        that are likely to change the policy.
        """
        # find all states that can reach each state
        predecessors = {}
        states = self.mdp.get_states()
        
        for state in states:
            predecessors[state] = set()
        
        for state in states:
            if self.mdp.is_terminal(state):
                continue
            possible_actions = self.mdp.get_possible_actions(state)
            if not possible_actions:
                continue
            for action in possible_actions:
                transitions = self.mdp.get_transition_states_and_probs(state, action)
                for next_state, prob in transitions:
                    if prob > 0:
                        predecessors[next_state].add(state)
        
        # init priority queue
        priority_queue = util.PriorityQueue()
        
        # add all states to queue with their error
        for state in states:
            if self.mdp.is_terminal(state):
                continue
            
            possible_actions = self.mdp.get_possible_actions(state)
            if not possible_actions:
                max_q_value = 0.0
            else:
                max_q_value = float('-inf')
                for action in possible_actions:
                    q_value = self.compute_q_value_from_values(state, action)
                    if q_value > max_q_value:
                        max_q_value = q_value
            
            # error = how wrong our current value is
            diff = abs(self.values[state] - max_q_value)
            priority_queue.put(state, -diff)
        
        # update states by priority
        for iteration in range(self.iterations):
            if priority_queue.is_empty():
                break
            
            state = priority_queue.get()
            
            if not self.mdp.is_terminal(state):
                possible_actions = self.mdp.get_possible_actions(state)
                if not possible_actions:
                    self.values[state] = 0.0
                else:
                    max_value = float('-inf')
                    for action in possible_actions:
                        q_value = self.compute_q_value_from_values(state, action)
                        if q_value > max_value:
                            max_value = q_value
                    self.values[state] = max_value
            
            # update predecessors if they need it
            for predecessor in predecessors[state]:
                if self.mdp.is_terminal(predecessor):
                    continue
                
                pred_actions = self.mdp.get_possible_actions(predecessor)
                if not pred_actions:
                    max_q_value = 0.0
                else:
                    max_q_value = float('-inf')
                    for action in pred_actions:
                        q_value = self.compute_q_value_from_values(predecessor, action)
                        if q_value > max_q_value:
                            max_q_value = q_value
                
                diff = abs(self.values[predecessor] - max_q_value)
                
                if diff > self.theta:
                    priority_queue.set(predecessor, -diff)