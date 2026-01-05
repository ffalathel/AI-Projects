# logicPlan.py
# ------------
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

"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

from typing import Dict, List, Tuple, Callable, Generator, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parse_expr, pl_true, TRUE

import itertools
import copy

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

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}


#______________________________________________________________________________
# QUESTION 1

def sentence1() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    
    #A ∨ B
    s1 = A | B
    
    #¬A ↔ (¬B ∨ C)
    s2 = ~A % (~B | C)
    
    #¬A ∨ ¬B ∨ C
    s3 = disjoin([~A, ~B, C])
    
    return conjoin([s1, s2, s3])
    "*** END YOUR CODE HERE ***"


def sentence2() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    D = Expr('D')
    
    #C ↔ (B ∨ D)
    s1 = C % (B | D)
    
    #A → (¬B ∧ ¬D)
    s2 = A >> (~B & ~D)
    
    #¬(B ∧ ¬C) → A
    s3 = ~(B & ~C) >> A
    
    #¬D → C
    s4 = ~D >> C
    
    
    return conjoin([s1, s2, s3, s4])
    "*** END YOUR CODE HERE ***"


def sentence3() -> Expr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.

    (Project update: for this question only, [0] and _t are both acceptable.)
    """
    "*** BEGIN YOUR CODE HERE ***"
    #Create symbols
    PacmanAlive_0 = PropSymbolExpr('PacmanAlive', time=0)
    PacmanAlive_1 = PropSymbolExpr('PacmanAlive', time=1)
    PacmanBorn_0 = PropSymbolExpr('PacmanBorn', time=0)
    PacmanKilled_0 = PropSymbolExpr('PacmanKilled', time=0)
    
    #PacmanAlive_1 ↔ ((PacmanAlive_0 ∧ ¬PacmanKilled_0) ∨ (¬PacmanAlive_0 ∧ PacmanBorn_0))
    s1 = PacmanAlive_1 % ((PacmanAlive_0 & ~PacmanKilled_0) | (~PacmanAlive_0 & PacmanBorn_0))
    
    #¬(PacmanAlive_0 ∧ PacmanBorn_0)
    s2 = ~(PacmanAlive_0 & PacmanBorn_0)
    
    #PacmanBorn_0
    s3 = PacmanBorn_0
    
    return conjoin([s1, s2, s3])
    "*** END YOUR CODE HERE ***"


def find_model(sentence: Expr) -> Dict[Expr, bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    cnf_sentence = to_cnf(sentence)
    return pycoSAT(cnf_sentence)

def find_model_check() -> Dict[Any, bool]:
    """Returns the result of find_model(Expr('a')) if lower cased expressions were allowed.
    You should not use find_model or Expr in this method.
    This can be solved with a one-line return statement.
    """
    class dummyClass:
        """dummy('A') has representation A, unlike a string 'A' that has repr 'A'.
        Of note: Expr('Name') has representation Name, not 'Name'.
        """
        def __init__(self, variable_name: str = 'A'):
            self.variable_name = variable_name
        
        def __repr__(self):
            return self.variable_name
    
    "*** BEGIN YOUR CODE HERE ***"
    #find_model(Expr('a')) would return {a: True} a is an Expr object. here return the same structure but with a dummyClass instance instead
    return {dummyClass('a'): True}
    "*** END YOUR CODE HERE ***"


def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """
    "*** BEGIN YOUR CODE HERE ***"
    #there is no model where premise is true and conclusion is false (premise ∧ ¬conclusion)
    test_sentence = premise & ~conclusion
    return find_model(test_sentence) == False
    "*** END YOUR CODE HERE ***"


def pl_true_inverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """
    "*** BEGIN YOUR CODE HERE ***"
    #check if ¬inverse_statement is true
    return pl_true(~inverse_statement, assignments)
    "*** END YOUR CODE HERE ***"


#______________________________________________________________________________
# QUESTION 2

def at_least_one(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals ist is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = at_least_one(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    #at least one means disjunction of all literals
    return disjoin(literals)
    "*** END YOUR CODE HERE ***"


def at_most_one(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """
    "*** BEGIN YOUR CODE HERE ***"
    #at most one means for every pair of literals, at least one is false
    clauses = []
    for lit1, lit2 in itertools.combinations(literals, 2):
        # At most one of lit1 and lit2 can be true ¬lit1 ∨ ¬lit2
        clauses.append(~lit1 | ~lit2)
    
    #if there are no clauses, it means there were 0 or 1 literals, so at most one is trivially true
    if len(clauses) == 0:
        return Expr('TRUE') 
    
    return conjoin(clauses)
    "*** END YOUR CODE HERE ***"


def exactly_one(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    #exactly one means at least one and at most one
    return conjoin([at_least_one(literals), at_most_one(literals)])
    "*** END YOUR CODE HERE ***"


#______________________________________________________________________________
# QUESTION 3

def pacman_successor_axiom_single(x: int, y: int, time: int, walls_grid: List[List[bool]]=None) -> Expr:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).

    Current <==> (previous position at time t-1) & (took action to move to x, y)

    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    
    Logic: P[x,y]_t ↔ (P[x,y+1]_t-1 ∧ South_t-1) ∨ (P[x,y-1]_t-1 ∧ North_t-1) ∨
                      (P[x+1,y]_t-1 ∧ West_t-1) ∨ (P[x-1,y]_t-1 ∧ East_t-1)
    """
    now, last = time, time - 1
    possible_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t
    
    #If last < 0, we can't have predecessor states, so return None
    if last < 0:
        return None
    
    #The if statements give a small performance boost and are required for q4 and q5 correctness
    if walls_grid[x][y+1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                                & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                                & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                                & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                                & PropSymbolExpr('East', time=last))

    if not possible_causes:
        return None
    
    "*** BEGIN YOUR CODE HERE ***"
    #P[x,y]_t ↔ (disjunction of all possible causes from previous timestep)
    return PropSymbolExpr(pacman_str, x, y, time=now) % disjoin(possible_causes)
    "*** END YOUR CODE HERE ***"


def slam_successor_axiom_single(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    """
    Similar to `pacman_successor_state_axioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    
    Logic: P[x,y]_t ↔ (P[x,y]_t-1 ∧ Stay_t-1) ∨ (P[x,y+1]_t-1 ∧ South_t-1) ∨
                      (P[x,y-1]_t-1 ∧ North_t-1) ∨ (P[x+1,y]_t-1 ∧ West_t-1) ∨ (P[x-1,y]_t-1 ∧ East_t-1)
    """
    now, last = time, time - 1
    moved_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    if walls_grid[x][y+1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))

    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(pacman_str, x, y, time=last), ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    failed_move_causes: List[Expr] = [] # using merged variables, improves speed significantly
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysics_axioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensor_model: Callable = None, successor_axioms: Callable = None) -> Expr:
    """
    Generates comprehensive physics axioms for timestep t.
    
    Logic includes:
    - ∀(x,y) ∈ all_coords: WALL[x,y] → ¬P[x,y]_t
    - exactly_one(P[x,y]_t for (x,y) ∈ non_outer_wall_coords)
    - exactly_one(Action_t for Action ∈ DIRECTIONS)
    - sensor_model axioms (if provided)
    - successor_axioms (if provided and t > 0)
    
    Args:
        t: timestep
        all_coords: all coordinates in the problem
        non_outer_wall_coords: coordinates excluding outer border
        walls_grid: wall locations for successor axioms
        sensor_model: function generating sensor axioms
        successor_axioms: function generating successor axioms
    """
    pacphysics_sentences = []

    "*** BEGIN YOUR CODE HERE ***"
    # ∀(x,y) ∈ all_coords: WALL[x,y] → ¬P[x,y]_t
    for x, y in all_coords:
        wall_implies_no_pacman = PropSymbolExpr(wall_str, x, y) >> ~PropSymbolExpr(pacman_str, x, y, time=t)
        pacphysics_sentences.append(wall_implies_no_pacman)
    
    #Exactly_one(P[x,y]_t for (x,y) ∈ non_outer_wall_coords)
    pacman_location_exprs = [PropSymbolExpr(pacman_str, x, y, time=t) for x, y in non_outer_wall_coords]
    pacphysics_sentences.append(exactly_one(pacman_location_exprs))
    
    #Exactly_one(Action_t for Action ∈ DIRECTIONS)
    action_exprs = [PropSymbolExpr(action, time=t) for action in DIRECTIONS]
    pacphysics_sentences.append(exactly_one(action_exprs))
    
    #Add sensor model axioms if provided
    if sensor_model is not None:
        pacphysics_sentences.append(sensor_model(t, non_outer_wall_coords))
    
    #Add successor axioms if provided and t > 0 (no predecessor states for t=0)
    if successor_axioms is not None and t > 0:
        pacphysics_sentences.append(successor_axioms(t, walls_grid, non_outer_wall_coords))
    
    "*** END YOUR CODE HERE ***"

    return conjoin(pacphysics_sentences)


def check_location_satisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Checks satisfiability of Pacman being at location (x1,y1) at time t=1.
    
    Logic: KB ∧ P[x1,y1]_1 and KB ∧ ¬P[x1,y1]_1
    Returns models for both cases to determine if location is possible/impossible.
    
    Args:
        x1_y1: potential location at time t=1
        x0_y0: Pacman's location at time t=0 
        action0: action taken at time t=0
        action1: action taken at time t=1
        problem: LocMapProblem instance with known walls
    
    Returns:
        (model1, model2): models for KB ∧ P[x1,y1]_1 and KB ∧ ¬P[x1,y1]_1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.as_list()
    all_coords = list(itertools.product(range(problem.get_width()+2), range(problem.get_height()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.get_width()+1), range(1, problem.get_height()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    "*** BEGIN YOUR CODE HERE ***"
    #Add pacphysics axioms for timestep 0 (no successor axioms since t-1 < 0)
    KB.append(pacphysics_axioms(0, all_coords, non_outer_wall_coords, walls_grid, 
                                sensor_model=None, 
                                successor_axioms=None))

    #Add pacphysics axioms for timestep 1 (with successor axioms)
    KB.append(pacphysics_axioms(1, all_coords, non_outer_wall_coords, walls_grid, 
                                sensor_model=None, 
                                successor_axioms=all_legal_successor_axioms))
    
    #Add initial Pacman location: P[x0,y0]_0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    
    #Add action knowledge: Action0_0 ∧ Action1_1
    KB.append(PropSymbolExpr(action0, time=0))
    KB.append(PropSymbolExpr(action1, time=1))
    
    #Conjoin all KB
    kb_conjoined = conjoin(KB)
    
    #Check satisfiability: KB ∧ P[x1,y1]_1
    model1 = find_model(kb_conjoined & PropSymbolExpr(pacman_str, x1, y1, time=1))
    
    #Check satisfiability: KB ∧ ¬P[x1,y1]_1
    model2 = find_model(kb_conjoined & ~PropSymbolExpr(pacman_str, x1, y1, time=1))
    
    return (model1, model2)
    "*** END YOUR CODE HERE ***"


#______________________________________________________________________________
# QUESTION 4

def position_logic_plan(problem) -> List:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysics_axioms.
    """
    walls_grid = problem.walls
    width, height = problem.get_width(), problem.get_height()
    walls_list = walls_grid.as_list()
    x0, y0 = problem.start_state
    xg, yg = problem.goal
    
    #Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
                                        range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    #pacman's initial location at timestep 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    
    #iterate through timesteps
    for t in range(50):
        # Print timestep for debugging
        print(f"Timestep {t}")
        
        #pacman can only be at exactly one of the non_wall_coords at timestep t
        pacman_location_exprs = [PropSymbolExpr(pacman_str, x, y, time=t) for x, y in non_wall_coords]
        KB.append(exactly_one(pacman_location_exprs))
        
        #pacman is at the goal at timestep t
        goal_assertion = PropSymbolExpr(pacman_str, xg, yg, time=t)
        
        #check if there's a satisfying assignment
        model = find_model(conjoin(KB) & goal_assertion)
        if model != False:
            #return the action sequence
            return extract_action_sequence(model, actions)
        
        #pacman takes exactly one action per timestep
        action_exprs = [PropSymbolExpr(action, time=t) for action in actions]
        KB.append(exactly_one(action_exprs))
        
        #for all possible positions add successor axioms
        for x, y in non_wall_coords:
            successor_axiom = pacman_successor_axiom_single(x, y, t+1, walls_grid)
            if successor_axiom:
                KB.append(successor_axiom)
    
    
    return []
    "*** END YOUR CODE HERE ***"


#______________________________________________________________________________
# QUESTION 5

def food_logic_plan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysics_axioms.
    """
    walls = problem.walls
    width, height = problem.get_width(), problem.get_height()
    walls_list = walls.as_list()
    (x0, y0), food = problem.start
    food = food.as_list()

    #Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    #pacman's initial location at timestep 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    
    #initialize food at timestep 0
    for x, y in food:
        KB.append(PropSymbolExpr(food_str, x, y, time=0))
    
    #iterate through timesteps
    for t in range(50):
        
        #pacman can only be at exactly one of the non_wall_coords at timestep t
        pacman_location_exprs = [PropSymbolExpr(pacman_str, x, y, time=t) for x, y in non_wall_coords]
        KB.append(exactly_one(pacman_location_exprs))
        
        #all food has been eaten 
        goal_assertion = conjoin([~PropSymbolExpr(food_str, x, y, time=t) for x, y in food])
        
        #check if there's a satisfying assignment
        model = find_model(conjoin(KB) & goal_assertion)
        if model != False:
            #return the action sequence
            return extract_action_sequence(model, actions)
        
        #pacman takes exactly one action per timestep
        action_exprs = [PropSymbolExpr(action, time=t) for action in actions]
        KB.append(exactly_one(action_exprs))
        
        #for all possible positions add successor axioms
        for x, y in non_wall_coords:
            successor_axiom = pacman_successor_axiom_single(x, y, t+1, walls)
            if successor_axiom:
                KB.append(successor_axiom)
        
        #add food successor axiom for each food location food[x,y]_t+1 ↔ (Food[x,y]_t ∧ ¬P[x,y]_t)
        for x, y in food:
            food_successor = PropSymbolExpr(food_str, x, y, time=t+1) % (
                PropSymbolExpr(food_str, x, y, time=t) & ~PropSymbolExpr(pacman_str, x, y, time=t)
            )
            KB.append(food_successor)
    
    
    return []
    "*** END YOUR CODE HERE ***"


#______________________________________________________________________________
# QUESTION 6

def localization(problem, agent) -> Generator:
    """
    Determines possible Pacman locations at each timestep using sensor readings.
    
    Logic: For each timestep t, find all locations (x,y) where KB ∧ P[x,y]_t is satisfiable.
    Uses 4-bit sensor readings to progressively narrow down possible locations.
    
    Args:
        problem: LocalizationProblem instance with known walls
        agent: LocalizationLogicAgent instance with sensor readings
    
    Yields:
        List of possible (x,y) locations at each timestep
    """
    walls_grid = problem.walls
    walls_list = walls_grid.as_list()
    all_coords = list(itertools.product(range(problem.get_width()+2), range(problem.get_height()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.get_width()+1), range(1, problem.get_height()+1)))

    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    #Add wall information: ∀(x,y) ∈ walls_list: WALL[x,y], ∀(x,y) ∉ walls_list: ¬WALL[x,y]
    for x, y in all_coords:
        if (x, y) in walls_list:
            KB.append(PropSymbolExpr(wall_str, x, y))
        else:
            KB.append(~PropSymbolExpr(wall_str, x, y))
    
    for t in range(agent.num_timesteps):
        #Add pacphysics, action, and percept information to KB
        KB.append(pacphysics_axioms(t, all_coords, non_outer_wall_coords, walls_grid,
                                    sensor_model=sensor_axioms,
                                    successor_axioms=all_legal_successor_axioms))
        
        #Add action taken at timestep t
        KB.append(PropSymbolExpr(agent.actions[t], time=t))
        
        #Get and add percept rules to KB
        percepts = agent.get_percepts()
        KB.append(four_bit_percept_rules(t, percepts))
        
        #Find possible Pacman locations with updated KB
        possible_locations = []
        kb_conjoined = conjoin(KB)
        
        for x, y in non_outer_wall_coords:
            pacman_at = PropSymbolExpr(pacman_str, x, y, time=t)
            
            # Check if KB ∧ P[x,y]_t is satisfiable (i.e., location is possible)
            model = find_model(kb_conjoined & pacman_at)
            
            if model != False:
                possible_locations.append((x, y))
        
        #Move agent to next state
        agent.move_to_next_state(agent.actions[t])
        
        yield possible_locations
    "*** END YOUR CODE HERE ***"


#______________________________________________________________________________
# QUESTION 7

def mapping(problem, agent) -> Generator:
    """
    Determines wall locations in an unknown map using sensor readings.
    
    Logic: For each timestep t, determine wall locations using KB ∧ WALL[x,y] and KB ∧ ¬WALL[x,y].
    Uses 4-bit sensor readings to progressively discover wall locations.
    
    Args:
        problem: MappingProblem instance with unknown walls
        agent: MappingLogicAgent instance with sensor readings
    
    Yields:
        known_map: 2D array where 1=wall, 0=non-wall, -1=unknown
    """
    pac_x_0, pac_y_0 = problem.start_state
    KB = []
    all_coords = list(itertools.product(range(problem.get_width()+2), range(problem.get_height()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.get_width()+1), range(1, problem.get_height()+1)))

    #map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.get_height()+2)] for x in range(problem.get_width()+2)]

    #Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.get_width() + 1)
                or (y == 0 or y == problem.get_height() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    #Add initial Pacman location: P[pac_x_0, pac_y_0]_0
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))

    #Pacman's starting location is not a wall: ¬WALL[pac_x_0, pac_y_0]
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))
    known_map[pac_x_0][pac_y_0] = 0
    
    for t in range(agent.num_timesteps):
        #Create walls grid with only known walls (1) and known non-walls (0)
        walls_grid = [[0 for y in range(problem.get_height()+2)] for x in range(problem.get_width()+2)]
        
        #Set outer walls
        for x, y in all_coords:
            if ((x == 0 or x == problem.get_width() + 1)
                    or (y == 0 or y == problem.get_height() + 1)):
                walls_grid[x][y] = 1
        
        #Set known walls and non-walls from known_map
        for x, y in non_outer_wall_coords:
            if known_map[x][y] == 1: # Known wall
                walls_grid[x][y] = 1
            elif known_map[x][y] == 0: # Known non-wall
                walls_grid[x][y] = 0
        
        #Add pacphysics, action, and percept information to KB
        KB.append(pacphysics_axioms(t, all_coords, non_outer_wall_coords, walls_grid,
                                    sensor_model=sensor_axioms,
                                    successor_axioms=all_legal_successor_axioms))
        
        #Add action taken at timestep t
        KB.append(PropSymbolExpr(agent.actions[t], time=t))
        
        #Get percepts and add percept rules to KB
        percepts = agent.get_percepts()
        KB.append(four_bit_percept_rules(t, percepts))
        
        #Find provable wall locations with updated KB
        kb_conjoined = conjoin(KB)
        
        for x, y in non_outer_wall_coords:
            wall_at = PropSymbolExpr(wall_str, x, y)
            
            #Can we prove there IS a wall at (x, y)?
            is_wall = entails(kb_conjoined, wall_at)
            
            #Can we prove there is NOT a wall at (x, y)?
            not_wall = entails(kb_conjoined, ~wall_at)
            
            #Update KB and known_map based on what we proved
            if is_wall:
                KB.append(wall_at)
                known_map[x][y] = 1
            elif not_wall:
                KB.append(~wall_at)
                known_map[x][y] = 0
            #If neither is_wall nor not_wall, keep known_map[x][y] as -1 (unknown)
        
        
        #Move agent to next state
        agent.move_to_next_state(agent.actions[t])
        
        yield known_map
    "*** END YOUR CODE HERE ***"


#______________________________________________________________________________
# QUESTION 8

def slam(problem, agent) -> Generator:
    """
    Simultaneous Localization and Mapping (SLAM) using sensor readings.
    
    Logic: Combines localization and mapping - determines both Pacman's location
    and wall locations simultaneously using KB ∧ P[x,y]_t and KB ∧ WALL[x,y].
    
    Args:
        problem: SLAMProblem instance with unknown walls and unknown starting location
        agent: SLAMLogicAgent instance with sensor readings
    
    Yields:
        (known_map, possible_locations): map and possible Pacman locations at each timestep
    """
    pac_x_0, pac_y_0 = problem.start_state
    KB = []
    all_coords = list(itertools.product(range(problem.get_width()+2), range(problem.get_height()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.get_width()+1), range(1, problem.get_height()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.get_height()+2)] for x in range(problem.get_width()+2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.get_width() + 1)
                or (y == 0 or y == problem.get_height() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    util.raiseNotDefined()

    for t in range(agent.num_timesteps):
        "*** END YOUR CODE HERE ***"
        yield (known_map, possible_locations)


# Abbreviations
plp = position_logic_plan
loc = localization
mp = mapping
flp = food_logic_plan

# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)


#______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.

def sensor_axioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time = t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def four_bit_percept_rules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], time=t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def num_adj_walls_percept_rules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker num_adj_walls_percept_rules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def slam_sensor_axioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def all_legal_successor_axioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """
    Generates successor axioms for all non-outer-wall coordinates at timestep t.
    
    Logic: ∧(x,y) ∈ non_outer_wall_coords: P[x,y]_t ↔ (possible causes from t-1)
    
    Args:
        t: timestep
        walls_grid: wall locations (2D array of ints or bools)
        non_outer_wall_coords: coordinates excluding outer border
    
    Returns:
        Conjunction of all successor axioms for valid coordinates
    """
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacman_successor_axiom_single(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def slam_successor_axioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """
    Generates SLAM successor axioms for all non-outer-wall coordinates at timestep t.
    
    Logic: ∧(x,y) ∈ non_outer_wall_coords: P[x,y]_t ↔ (Stay_t-1 ∨ possible moves from t-1)
    
    Args:
        t: timestep
        walls_grid: wall locations (2D array of ints or bools)
        non_outer_wall_coords: coordinates excluding outer border
    
    Returns:
        Conjunction of all SLAM successor axioms for valid coordinates
    """
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = slam_successor_axiom_single(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


#______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def model_to_string(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def extract_action_sequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extract_action_sequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parse_expr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualize_coords(coords_list, problem) -> None:
    wall_grid = game.Grid(problem.walls.width, problem.walls.height, initial_value=False)
    for (x, y) in itertools.product(range(problem.get_width()+2), range(problem.get_height()+2)):
        if (x, y) in coords_list:
            wall_grid.data[x][y] = True
    print(wall_grid)


# Helpful Debug Method
def visualize_bool_array(bool_arr, problem) -> None:
    wall_grid = game.Grid(problem.walls.width, problem.walls.height, initial_value=False)
    wall_grid.data = copy.deepcopy(bool_arr)
    print(wall_grid)


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def get_ghost_start_states(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
    
    def get_goal_state(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()