# Artificial Intelligence Project Portfolio

This repository contains five core projects completed as part of the Introduction to AI curriculum at USF. These projects progress from classical search and logic to modern reinforcement learning and LLM integration.

---

## Table of Contents

1. [Pacman Search](#project-1-pacman-search)
2. [Multi-Agent Search](#project-2-multi-agent-search)
3. [Logic and Classical Planning](#project-3-logic-and-classical-planning)
4. [Reinforcement Learning](#project-4-reinforcement-learning)
5. [ML, DL, and LLMs](#project-5-ml-dl-and-llms)

---

## Project 1: Pacman Search

### Overview

In this project, I developed a set of general search algorithms and applied them to various Pacman scenarios. The goal was to program a Pacman agent that can navigate its maze world efficiently, both to reach specific locations and to collect food dots.

This project demonstrates the application of classic AI search techniques to pathfinding and state-space exploration.

### Implemented Search Algorithms

| Algorithm | Description |
|-----------|-------------|
| Depth-First Search (DFS) | Implemented using a LIFO data structure to find paths in various maze layouts |
| Breadth-First Search (BFS) | Implemented using a FIFO data structure to ensure least-cost solutions |
| Uniform-Cost Search (UCS) | Designed to find the "best" path by varying cost functions for different terrains or risks |
| A* Search | Utilized heuristic functions (like Manhattan distance) to find optimal paths more efficiently than UCS |

### Advanced Problem Solving

Beyond basic pathfinding, I tackled more complex search problems:

**Corners Problem** — Formulated a search problem and a consistent heuristic to find the shortest path through all four corners of a maze.

**Food Search Problem** — Designed a consistent heuristic to solve the "all-dots" food-clearing problem.

**Suboptimal Greedy Search** — Implemented a ClosestDotSearchAgent that greedily navigates to the nearest food source.

### Files & Structure

| File | Purpose |
|------|---------|
| `search.py` | Contains all generic search algorithms |
| `searchAgents.py` | Contains search-based agents and problem definitions |
| `pacman.py` | The main game engine |
| `util.py` | Supporting data structures (Priority Queues, Stacks, Queues) |

### Usage

```bash
# Run Pacman
python pacman.py

# Run autograder
python autograder.py
```

### Engineering Process

During development, I prioritized:

- **Algorithmic Consistency** — Ensuring heuristics remained admissible and consistent to guarantee optimal solutions in A*
- **State Representation** — Defining abstract states for complex problems (like the Corners Problem) that excluded irrelevant game data to maintain performance
- **Efficiency** — Aiming for minimal node expansion to meet the autograder's high-performance benchmarks

---

## Project 2: Multi-Agent Search

### Overview

In this project, I designed intelligent agents for a classic version of Pacman that includes adversarial ghosts. Unlike the first project, which focused on pathfinding in static environments, this project involved decision-making in competitive environments where the state of the world changes based on the actions of other agents.

I implemented various adversarial search algorithms to allow Pacman to navigate mazes while actively avoiding or outsmarting ghosts.

### Implemented Search Agents

| Agent | Description |
|-------|-------------|
| Reflex Agent | Evaluation function considering food locations and ghost proximity for immediate decisions without deep look-ahead |
| Minimax Agent | Classic Minimax algorithm generalized to handle multiple adversary (ghost) layers for every max (Pacman) layer |
| Alpha-Beta Pruning | Optimized Minimax search that significantly reduces nodes expanded, allowing deeper search trees |
| Expectimax Agent | Models probabilistic behavior for better performance against suboptimal or random ghost movements |
| Better Evaluation Function | State-evaluation accounting for distance to nearest food, ghost "scared" timers, and maze congestion |

### Key Files

| File | Purpose |
|------|---------|
| `multiAgents.py` | Primary file containing all multi-agent search logic |
| `pacman.py` | Main game engine and GameState definitions |
| `ghostAgents.py` | Logic for ghost movement (including DirectionalGhost) |

### Usage

```bash
# Run a specific agent
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic

# Run autograder
python autograder.py
```

### Engineering Process

My development focused on:

- **Generalization** — Ensuring the Minimax and Expectimax algorithms could scale to any number of ghosts
- **Performance Optimization** — Carefully implementing Alpha-Beta pruning to ensure no unnecessary `generate_successor` calls were made
- **Feature Engineering** — Iteratively refining the `betterEvaluationFunction` to balance aggressive food-gathering with defensive positioning

---

## Project 3: Logic and Classical Planning

### Overview

In this project, I implemented logical agents that use propositional logic to solve tasks in the Pacman world. By constructing logical sentences that describe "pacphysics," I utilized a SAT solver (pycosat) to handle complex inference tasks. This approach allowed Pacman to navigate, plan, and understand its environment through logical deduction rather than state-space search alone.

### Key Inference Tasks

| Task | Description |
|------|-------------|
| Logical Planning | Generating action sequences to reach goal locations and consume all food dots by asserting goal states in the knowledge base |
| Localization | Identifying Pacman's possible locations within a known map based on a local sensor model and action history |
| Mapping | Building an internal map of the environment by deducing wall locations based on percepts |
| SLAM | Simultaneous Localization and Mapping — finding Pacman's location while constructing a map of an unknown environment |

### Logic Implementation

The core of this project involved building a robust Knowledge Base (KB) using:

**Successor State Axioms** — Defining how the state of the world (Pacman's position, food presence) changes over time based on specific actions.

**Pacphysics** — Creating logical expressions that define physical rules of the maze, such as "Pacman cannot be in a wall" and "Pacman can only be in one location at a time."

**Satisfiability (SAT) Solving** — Using the `find_model` function to query the SAT solver for satisfying assignments that represent valid paths or environment states.

### Files & Structure

| File | Purpose |
|------|---------|
| `logicPlan.py` | Primary file with logic-based agents and planning functions |
| `logic.py` | Defines the `Expr` class for propositional logic sentences (AND, OR, NOT, Implication, Biconditional) |
| `logicAgents.py` | Definitions for specific logical problems Pacman encounters |
| `pycosat` | External library for solving CNF logic problems |

### Usage

```bash
# Run Planning
python pacman.py -l tinyMaze -p LogicAgent -a fn=plp

# Run Localization
python autograder.py -q q6

# Run Mapping
python autograder.py -q q7
```

### Engineering Process

The development process centered on:

- **Propositional Modeling** — Translating English descriptions of game physics into precise logical symbols and operators
- **CNF Efficiency** — Utilizing `conjoin` and `disjoin` to maintain flat logic trees, ensuring the SAT solver could process deep timesteps (up to 50 steps) without exponential slowdown
- **Axiomatic Design** — Ensuring successor state axioms were both necessary and sufficient to prevent deducing "impossible" states

---

## Project 4: Reinforcement Learning

### Overview

In this project, I implemented several reinforcement learning agents to solve Markov Decision Processes (MDPs). The project progressed from offline value iteration to online Q-learning, testing these agents in Gridworld environments, a simulated robot crawler, and eventually a complex Pacman environment. The goal was to create agents that learn optimal policies through experience and trial-and-error.

### Offline Planning

| Algorithm | Description |
|-----------|-------------|
| Value Iteration | Standard offline MDP solver that computes optimal values and policies before environment interaction |
| Asynchronous Value Iteration | More efficient version updating one state at a time in cyclic manner rather than batch updates |
| Prioritized Sweeping | Optimized planning using a priority queue to focus updates on states with highest error |

### Online Learning

| Algorithm | Description |
|-----------|-------------|
| Q-Learning | Agent learns by trial and error, updating Q-values based on experienced rewards and transitions |
| Epsilon-Greedy Exploration | Balances exploiting known high-value actions with exploring new actions |
| Approximate Q-Learning | Scalable learning for large state spaces using feature extraction to generalize across similar states |

### Key Files

| File | Purpose |
|------|---------|
| `valueIterationAgents.py` | Contains Value Iteration and Prioritized Sweeping agents |
| `qlearningAgents.py` | Contains basic Q-learning and Approximate Q-learning agents |
| `analysis.py` | Parameter tuning (discount, noise, epsilon) for specific environment challenges |
| `featureExtractors.py` | Feature definitions used by Approximate Q-learner |

### Testing Applications

| Environment | Purpose |
|-------------|---------|
| Gridworld | Visualize value convergence and test exploration/exploitation trade-offs |
| Crawler | Apply Q-learning to a simulated robot learning a physical crawling gait |
| Pacman | Utilize Approximate Q-learning on large layouts by learning features rather than individual configurations |

### Engineering Process

My approach focused on:

- **Generalization** — Writing Q-learning logic that is environment-agnostic, working seamlessly for Gridworld, the Crawler, and Pacman
- **Efficiency** — Implementing Prioritized Sweeping to drastically reduce updates needed for value convergence
- **Feature Engineering** — Utilizing the `SimpleExtractor` to help the Approximate Q-learner generalize patterns for larger Pacman maps

---

## Project 5: ML, DL, and LLMs

### Overview

This project shifted away from the Pacman environment to explore modern Artificial Intelligence through Machine Learning (ML), Deep Learning (DL), and Large Language Models (LLMs). The work was split into two primary phases: evaluating traditional and deep learning models on facial recognition and programmatically interacting with LLMs for text classification and reasoning.

### Phase 1: Machine Learning & Deep Learning

In this phase, I implemented and evaluated various models on a facial recognition dataset to compare their predictive capabilities.

**Model Implementation** — Ran several ML and DL methods to identify patterns in facial data.

**Performance Evaluation** — Evaluated models based on accuracy and efficiency, practicing the iterative process of training and hyperparameter selection.

### Phase 2: Large Language Models

This phase focused on programmatic interaction with LLMs and comparing their performance against deep learning baselines.

**Prompt Engineering** — Tested different prompting methods to evaluate how variations in instructions affect LLM behavior and output quality.

**Programmatic LLM Interaction** — Used Python to interface with LLMs, managing the challenges of running these models on CPU-limited environments.

**Baseline Comparison** — Compared the qualitative and quantitative performance of LLMs against the deep learning models developed in Phase 1.

---

## Technologies Used

- Python
- pycosat (SAT Solver)
- NumPy
- Machine Learning / Deep Learning frameworks

---

## Acknowledgments

- UC Berkeley AI Division for the Pacman framework
- USF Introduction to AI course
