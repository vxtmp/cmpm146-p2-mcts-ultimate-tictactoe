
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

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
    #part inspired by ChatGPT
    while not board.is_ended(curr_state):
        # action = choice(board.legal_actions(curr_state))
        # curr_state = board.next_state(curr_state, action)
        actions = board.legal_actions(state)

        best_score = float("-inf")
        best_action = None
        for act in actions:
            score = evaluate_heuristic(board, curr_state, act)
            if(score > best_score):
                best_score = score
                best_action = act
        action = best_action
        curr_state = board.next_state(curr_state, action)


    return curr_state

def evaluate_heuristic(board, state, action, bot_id):
    """
    Evaluate the value of taking 'action' from 'state' for 'bot_id' (1 or 2).
    Larger positive values indicate a stronger move for 'bot_id'.
    """

    # Apply the action to get the resulting state.
    next_state = board.next_state(state, action)
    
    # 1. If this move ends the game, check if it's a win/loss/draw.
    if board.is_ended(next_state):
        outcome = board.points_values(next_state)  # e.g. {1: 1, 2: -1} if player 1 wins
        if outcome[bot_id] == 1:
            # Bot wins immediately => very high score
            return 10000.0
        elif outcome[bot_id] == -1:
            # Bot loses immediately => very negative
            return -10000.0
        else:
            # Draw => neutral
            return 0.0

    # 2. Score capturing local boards: compare owned boxes before & after.
    curr_owned = board.owned_boxes(state)
    new_owned = board.owned_boxes(next_state)

    # +50 points for each new local board the bot captures with this move,
    # -50 for each new local board the opponent captures.
    capture_score = 0.0
    for loc in new_owned:
        if curr_owned[loc] == 0 and new_owned[loc] == bot_id:
            capture_score += 50.0
        elif curr_owned[loc] == 0 and new_owned[loc] != 0 and new_owned[loc] != bot_id:
            capture_score -= 50.0
    
    # 3. Score macro-board alignment. If capturing a local board helps us form 
    # a row/col/diag on the macro board, that’s extra valuable.
    macro_score = score_macro_board(board, next_state, bot_id)

    # 4. (Optional) Score partial progress in each local board. 
    # e.g., 2 in a row with an empty third cell => +5, blocking opponent => +5, etc.
    micro_threat_score = score_micro_threats(board, next_state, bot_id)

    # Combine scores into a single value. You can tune weights.
    score = capture_score + macro_score + micro_threat_score
    return score


def score_macro_board(board, state, bot_id):
    """
    Scores how good the 'macro' board (3x3 of local boards) is for 'bot_id'.
    For instance, +20 for each 2/3 owned local boards in a line, +100 for each 3/3.
    """
    owned = board.owned_boxes(state)
    # owned[(row, col)] in {0, 1, 2} indicates who owns that local board

    # Convert from local board ownership to a 3x3 array for easier checking
    macro = [[owned[(r, c)] for c in range(3)] for r in range(3)]

    score = 0.0
    lines = []

    # Rows
    lines.extend([ (macro[r][0], macro[r][1], macro[r][2]) for r in range(3) ])
    # Columns
    lines.extend([ (macro[0][c], macro[1][c], macro[2][c]) for c in range(3) ])
    # Diagonals
    lines.append((macro[0][0], macro[1][1], macro[2][2]))
    lines.append((macro[0][2], macro[1][1], macro[2][0]))

    for line in lines:
        num_me = line.count(bot_id)
        num_opp = line.count(3 - bot_id)  # If bot_id=1, opp=2; if bot_id=2, opp=1

        # If the line is still open (the opponent hasn’t fully captured it):
        if num_opp == 0:
            if num_me == 2:
                # 2 out of 3 local boards are mine => good progress
                score += 20
            elif num_me == 3:
                # That might actually be a game-ending scenario, but let's add a big number
                score += 100

        # Maybe also penalize lines where the opponent has 2
        if num_me == 0 and num_opp == 2:
            score -= 20

    return score


def score_micro_threats(board, state, bot_id):
    """
    Score partial progress in local boards themselves.
    E.g., +5 for each 2-in-a-row with the 3rd space open,
    -5 if the opponent has 2-in-a-row. 
    """
    # This is more involved, since you'd need to read each local board from 'state'.
    # The Board class might have a method to retrieve the local boards or the full 9x9 layout.
    score = 0.0
    # For demonstration, let's just return 0.0 or show a placeholder approach.
    return score

#reasonable quick solution without finding a perfect solution
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
        node, state = traverse_nodes(node, board, state, bot_identity) # select. uses ucb
        
        if node != None:
            if node.untried_actions:
                node, state = expand_leaf(node, board, state) # expand
        
        state = rollout(board, state) # rollout
        won = is_win(board, state, bot_identity)
        
        backpropagate(node, won) # backpropagate

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node) # get best action. calculates win rate (not ucb).
    
    # print(f"Action chosen: {best_action}")
    return best_action
