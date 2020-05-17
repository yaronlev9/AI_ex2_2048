import numpy as np
import abc
import util
from game import Agent, Action
import copy
import math


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        # Useful information you can extract from a GameState (game_state.py)
        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score
        counter = get_adjacencies(board)
        new_board = down_matrix(board)
        counter += get_adjacencies(new_board)
        return max_tile + score + counter


def get_adjacencies(board):
    counter = 0
    prev_num = None
    for row in board:
        for cell in row:
            if (cell != 0 and prev_num == None):
                prev_num = cell
                continue
            if cell != 0:
                if cell == prev_num:
                    counter += 1
                    prev_num = None
                else:
                    prev_num = cell
        prev_num = None
    return counter


def down_matrix(matrix):
    """this function receives a matrix (two dimensional list), and returns
    it in the form of a two dimensional list, suited for words that
    might appear in the downward direction"""
    new_lst = []
    for i in range(len(matrix[0])):
        new_lst.append([])
        for lst in matrix:
            new_lst[i].append(lst[i])
    return new_lst


def get_diagonal(diagonal_matrix, matrix):
    """this function receives a matrix and an empty list and returns a
    two dimensional list suited for one direction of diagonals"""
    new_matrix = diagonal_matrix
    if len(matrix) == 0:
        return new_matrix
    else:
        for amount_of_diagonals in range(len(matrix) + len(matrix[0]) - 1):
            new_matrix.append([])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                new_matrix[j + i].append(matrix[i][j])


def up_right_diagonal(matrix):
    """this function receives a matrix (two dimensional list), and returns
    it in the form of a two dimensional list, suited for words that
    might appear in the up right diagonal direction"""
    new_matrix = copy.deepcopy(down_matrix(matrix))
    diagonal_matrix = []
    get_diagonal(diagonal_matrix, new_matrix)
    return diagonal_matrix


def monotonous_evaluation(board):
    result = 1
    diagonals_matrix = up_right_diagonal(copy.deepcopy(board))
    diagonals_matrix.reverse()
    last_min = diagonals_matrix[len(diagonals_matrix) - 1][0]
    for lst in diagonals_matrix:
        cur_min = last_min
        for element in lst:
            if element <= last_min:
                result += 1
            if element < cur_min:
                cur_min = element
        last_min = cur_min
    return result


def get_initialized_neighbors(board, row, col, penalized_set):
    result = []
    if col - 1 > 0 and board[row][col - 1] != 0 and (row, col - 1) not in penalized_set:
        result.append(board[row][col - 1])
    if row - 1 > 0 and board[row - 1][col] != 0 and (row - 1, col) not in penalized_set:
        result.append(board[row - 1][col])
    if col + 1 <= len(board[0]) - 1 and board[row][col + 1] != 0 and (row, col + 1) not in penalized_set:
        result.append(board[row][col + 1])
    if row + 1 <= len(board) - 1 and board[row + 1][col] != 0 and (row + 1, col) not in penalized_set:
        result.append(board[row + 1][col])
    return result


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def mini_max(self, game_state, depth, current_agent):
        current_state = game_state
        actions = game_state.get_legal_actions(current_agent)
        if depth == 0 or actions == []:
            return self.evaluation_function(current_state), None
        if current_agent == 0:  # if current is max player
            value = - math.inf
            for action in actions:
                successor = current_state.generate_successor(current_agent, action)
                value = max(value, (self.mini_max(successor, depth - 1, 1))[0])
                return value, action
        else:  # if current is min player
            value = math.inf
            for action in actions:
                successor = current_state.generate_successor(current_agent, action)
                value = min(value, (self.mini_max(successor, depth - 1, 0))[0])
                return value, action

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        mini_max_result = self.mini_max(game_state, self.depth, 0)
        print(mini_max_result[0])
        return mini_max_result[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
