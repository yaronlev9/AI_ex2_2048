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
    """
    gets the number of all the foldable tiles on the board.
    """
    counter = 0
    prev_num = None
    for row in board:
        for cell in row:
            if cell != 0 and prev_num is None:
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
    """receives a matrix (two dimensional list), and returns
    it in the form of a two dimensional list, suited for words that
    might appear in the downward direction"""
    new_lst = []
    for i in range(len(matrix[0])):
        new_lst.append([])
        for lst in matrix:
            new_lst[i].append(lst[i])
    return new_lst


def get_diagonal(diagonal_matrix, matrix):
    """receives a matrix and an empty list and returns a
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
    """receives a matrix (two dimensional list), and returns
    it in the form of a two dimensional list, suited for words that
    might appear in the up right diagonal direction"""
    new_matrix = copy.deepcopy(down_matrix(matrix))
    diagonal_matrix = []
    get_diagonal(diagonal_matrix, new_matrix)
    return diagonal_matrix


def monotonous_evaluation(board):
    """
    gets an evaluation of how monotonous is the board, considering the down right
    corner to be the highest value, and the diagonals adjacent to it as lower values
    in monotonous manner.
    """
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


def get_penalty(board, row, col, penalized_set):
    """
    gets an evaluation of the penalty to set upon a board state, by checking how many neighbor tiles are on
    the board that cannot be folded.
    """
    result = 0
    if col - 1 > 0 and board[row][col - 1] != 0 and (row, col - 1) not in penalized_set and board[row][col] != \
            board[row][col - 1]:
        result += 1
    if row - 1 > 0 and board[row - 1][col] != 0 and (row - 1, col) not in penalized_set and board[row][col] != \
            board[row - 1][col]:
        result += 1
    if col + 1 <= len(board[0]) - 1 and board[row][col + 1] != 0 and (row, col + 1) not in penalized_set and \
            board[row][col] != board[row][col + 1]:
        result += 1
    if row + 1 <= len(board) - 1 and board[row + 1][col] != 0 and (row + 1, col) not in penalized_set and \
            board[row][col] != board[row + 1][col]:
        result += 1
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
        """
        an implementation of the minimax algorithm.
        :param game_state: the beginning state of he game.
        :param depth: the maximum depth of the game tree.
        :param current_agent: 1 for max player, and 0 for min player.
        :return: the value of the state.
        """
        current_state = game_state
        actions = game_state.get_legal_actions(current_agent)
        if depth == 0 or actions == []:
            return self.evaluation_function(current_state)
        if current_agent == 0:  # if current is max player
            value = - math.inf
            for action in actions:
                successor = current_state.generate_successor(current_agent, action)
                value = max(value, (self.mini_max(successor, depth - 1, 1)))
            return value
        else:  # if current is min player
            value = math.inf
            for action in actions:
                successor = current_state.generate_successor(current_agent, action)
                value = min(value, (self.mini_max(successor, depth - 1, 0)))
            return value

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
        lst = []
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            lst.append([action, self.mini_max(successor, (self.depth * 2) - 1, 1)])
        lst.sort(key=lambda x: x[1])
        return lst[-1][0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alpha_beta_pruning(self, depth, game_state, alpha, beta, current_agent):
        """
        an implementation of alpha-beta pruning algorithm.
        """
        current_state = game_state
        actions = game_state.get_legal_actions(current_agent)
        if depth == 0 or actions == []:
            return self.evaluation_function(current_state)
        if current_agent == 0:  # if current is max player
            max_value = - math.inf
            for action in actions:
                successor = current_state.generate_successor(current_agent, action)
                value = self.alpha_beta_pruning(depth - 1, successor, alpha, beta, 1)
                max_value = max(max_value, value)
                alpha = max(value, alpha)
                if alpha >= beta:
                    break
            return max_value
        else:  # if current is min player
            min_value = math.inf
            for action in actions:
                successor = current_state.generate_successor(current_agent, action)
                value = self.alpha_beta_pruning(depth - 1, successor, alpha, beta, 0)
                min_value = min(min_value, value)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return min_value

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        lst = []
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            lst.append((action, self.alpha_beta_pruning(((self.depth * 2) - 1), successor, -math.inf, math.inf, 1)))
        lst.sort(key=lambda x: x[1])
        return lst[-1][0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def expectimax(self, game_state, depth, current_agent):
        """an implementation of the expectimax algorithm"""
        current_state = game_state
        actions = game_state.get_legal_actions(current_agent)
        if depth == 0 or actions == []:
            return self.evaluation_function(current_state)
        if current_agent == 0:  # if current is max player
            max_value = - math.inf
            for action in actions:
                successor = current_state.generate_successor(current_agent, action)
                value = self.expectimax(successor, depth - 1, 1)
                max_value = max(max_value, value)
            return max_value
        if current_agent == 1:  # if current is min player
            value = 0
            for action in actions:
                successor = current_state.generate_successor(current_agent, action)
                value += self.expectimax(successor, depth - 1, 0)
            value /= len(actions)
            return value

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        lst = []
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            lst.append((action, self.expectimax(successor, (self.depth * 2) - 1, 1)))
        lst.sort(key=lambda x: x[1])
        return lst[-1][0]


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    board = current_game_state.board
    max_tile = current_game_state.max_tile
    score = current_game_state.score
    counter = get_adjacencies(board)
    new_board = down_matrix(board)
    counter += get_adjacencies(new_board)
    num_of_vacancies = 0
    penalized_set = set()
    penalty = 1
    corner_multiplier = 1
    second_max = 0
    sum_lu_corners = 4 * (board[0][0]) + (board[0][1] + board[1][0]) + 0.5 * (board[1][1])
    sum_rd_corners = 3 * (board[3][3]) + 2 * (board[2][3] + board[3][2]) + 0.5 * (board[2][2])
    sum_ru_corners = 3 * (board[0][3]) + 2 * (board[0][2] + board[1][3]) + 0.5 * (board[1][2])
    sum_ld_corners = 3 * (board[3][0]) + 2 * (board[2][0] + board[3][1]) + 0.5 * (board[2][1])
    sum_corners = max(sum_ld_corners, sum_lu_corners, sum_rd_corners, sum_ru_corners)
    for row_index in range(len(board)):
        for col_index in range(len(board[0])):
            if board[row_index][col_index] == 0:
                num_of_vacancies += 1
            else:
                penalty += get_penalty(board, row_index, col_index, penalized_set)
                penalized_set.add((row_index, col_index))
            if max_tile > board[row_index][col_index] > second_max:
                second_max = board[row_index][col_index]
    if board[len(board) - 1][len(board[0]) - 1] == max_tile:
        corner_multiplier += 1
    if len(board) > 1:
        if board[len(board) - 2][len(board[0]) - 1] == second_max or board[len(board) - 2][
            len(board[0]) - 1] == max_tile:
            corner_multiplier += 1
        if board[len(board) - 1][len(board[0]) - 2] == second_max or board[len(board) - 2][
            len(board[0]) - 1] == max_tile:
            corner_multiplier += 1
    return max_tile + score + counter * 5 + sum_corners + num_of_vacancies - penalty


# Abbreviation
better = better_evaluation_function
