
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log
from time import time 

use_time_budget = True # True to use time budget, False to use node count budget
report_node_count = False # True to report node count in the tree and player ID for Experiment 3 Traverse tree to count nodes
time_budget = 1.0 # Time allowed to build MCTS tree in seconds. Only used if use_time_budget is True
num_nodes = 1000
explore_faction = 2.

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    current_node = node
    current_state = state
    
    if board.is_ended(state):
        return current_node, current_state
    
    # any untried actions left?
    if current_node.untried_actions:
        return current_node, current_state # is leaf node
    
    # 1 if the last action was performed by the opponent, 0 otherwise
    is_opponent = board.current_player(current_state) != bot_identity
    
    highest_ucb = -1
    best_state = None
    best_child = None
    # get best action and expand
    for action, child in current_node.child_nodes.items():
        child_ucb = ucb (child, is_opponent)
        if child_ucb == float('inf'):
            return child, board.next_state(current_state, action)
        if child_ucb > highest_ucb:
            highest_ucb = child_ucb
            best_state = board.next_state(current_state, action)
            best_child = child
    # recurse
    return traverse_nodes(best_child, board, best_state, bot_identity)

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    # expand the node by adding a new child node
    action = node.untried_actions.pop() # get untried action. remove from node's untried list.
    new_state = board.next_state(state, action)
    new_node = MCTSNode(parent=node, parent_action=action, action_list=board.legal_actions(new_state))
    node.child_nodes[action] = new_node
    return new_node, new_state

def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    # rollout randomly
    curr_state = state # duplicate state
    
    while not board.is_ended(curr_state):
        action = choice(board.legal_actions(curr_state))
        curr_state = board.next_state(curr_state, action)
    return curr_state

def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    # backpropagate the Monte Carlo return value
    if node is None:
        return
    node.visits += 1
    if won:
        node.wins += 1
    backpropagate(node.parent, won)

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calculates the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    # calculate the UCB value
    if node.visits == 0:
        return float('inf')
    exploration = explore_faction * sqrt( 
                                        log(node.parent.visits) / node.visits 
                                        )
    exploitation = node.wins / node.visits
    
    if is_opponent:
        exploitation = 1 - exploitation
        
    return exploitation + exploration
        

def get_best_action(root_node: MCTSNode, is_opponent: bool = False):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    # choose the child node with best UCB value
    best_action = None
    best_win_rate = -1
    for action, child in root_node.child_nodes.items():
        curr_win_rate = child.wins / child.visits
        if curr_win_rate > best_win_rate:
            best_win_rate = curr_win_rate
            best_action = action
    return best_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def count_nodes(node: MCTSNode):
    """ Traverses the tree and counts the number of nodes in the tree

    Args:
        node:   A tree node from which the search is traversing.

    Returns:
        The number of nodes in the tree

    """
    if not node.child_nodes:
        return 1
    else:
        count = 1
        for child in node.child_nodes.values():
            count += count_nodes(child)
        return count

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    if use_time_budget:
        start_time = time()
        while time() - start_time < time_budget:
            state = current_state
            node = root_node

            # Do MCTS
            node, state = traverse_nodes(node, board, state, bot_identity) # select. uses ucb
            
            if node != None:
                if node.untried_actions:
                    node, state = expand_leaf(node, board, state) # expand
            
            state = rollout(board, state) # rollout
            won = is_win(board, state, bot_identity)
            
            backpropagate(node, won) # backpropagate
    #else if not using time budget
    elif not use_time_budget:
        for _ in range(num_nodes):
            state = current_state
            node = root_node

            # Do MCTS
            node, state = traverse_nodes(node, board, state, bot_identity) # select. uses ucb
            
            if node != None:
                if node.untried_actions:
                    node, state = expand_leaf(node, board, state) # expand
            
            state = rollout(board, state) # rollout
            won = is_win(board, state, bot_identity)
            
            backpropagate(node, won) # backpropagate
            
    # report node count in the tree and player ID for Experiment 3 Traverse tree to count nodes
    if report_node_count:
        nodes_in_tree = 0
        nodes_in_tree = count_nodes(root_node)            
        print(f"Node count: {nodes_in_tree}, Player ID: {bot_identity}")

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node) # get best action. calculates win rate (not ucb).
    
    # print(f"Action chosen: {best_action}")
    return best_action
