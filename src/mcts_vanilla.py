
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 100
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
    while not board.is_ended(state):               # take a state, board checks if terminal.
        if node.untried_actions:
            return expand_leaf(node, board, state)
        else:
            node = get_best_action(node)
            state = board.next_state(state, node.parent_action)
    pass

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
    action = node.untried_actions.pop()
    new_state = board.next_state(state, action)
    new_node = MCTSNode(parent=node, parent_action=action, action_list=board.legal_actions(new_state))
    node.child_nodes[action] = new_node
    return new_node, new_state
    pass


def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    # rollout randomly
    while not board.is_ended(state):
        action = choice(board.legal_actions(state))
        state = board.next_state(state, action)
    return state
    pass


def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    # backpropagate the Monte Carlo return value
    node.visits += 1
    if won:
        node.wins += 1
    if node.parent is not None:
        backpropagate(node.parent, won)
    pass

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
    
    exploration = explore_faction * sqrt(log(node.parent.visits) / node.visits)
    exploitation = node.wins / node.visits
    ucb_value = exploitation + exploration
    
    if is_opponent:
        return 1.0 - ucb_value # prefer good opponent moves (lower ucb_value)
    else:
        return ucb_value

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    # choose the child node with best UCB value
    best_action = None
    best_value = -1
    for action, child in root_node.child_nodes.items():
        value = child.wins / child.visits + explore_faction * sqrt(log(root_node.visits) / child.visits)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node

        # Do MCTS
        node = traverse_nodes(node, board, state, bot_identity) # select
        
        node, state = expand_leaf(node, board, state) # expand. returns new node and new state
        
        state = rollout(board, state) # rollout
        won = is_win(board, state, bot_identity)
        
        backpropagate(node, won) # backpropagate

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)
    
    print(f"Action chosen: {best_action}")
    return best_action
