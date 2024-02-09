from textwrap import dedent
from math import log, sqrt
from copy import deepcopy
import pandasql as ps
from pick import pick
import pandas as pd
import numpy as np
import datetime
import random
import pygame
import time

#==================================================

WHITE = (255,255,255)
BROWN = (255,228,196)
GREEN = ( 34,139, 34)
BLACK = (  0,  0,  0)
RED   = (255,  0,  0)
PINK =  (255, 16,240)

max_depth = 3
delay_time = 3000
print_log = True
monte_log = True
test = True

tournaments = 5
number_of_games = 100

Dict = {}
Dict[0]  = ['RANDOM','RANDOM']
Dict[1]  = ['MINMAX','MINMAX']
Dict[2]  = ['MINMAX ALPHA-BETA PRUNING','MINMAX ALPHA-BETA PRUNING']
Dict[3]  = ['MONTE','MONTE']
Dict[4]  = ['HUMAN','HUMAN']
Dict[5]  = ['CONSOLE','CONSOLE']
Dict[6]  = ['RANDOM','MINMAX']
Dict[7]  = ['RANDOM','MINMAX ALPHA-BETA PRUNING']
Dict[8]  = ['RANDOM','MONTE']
Dict[9]  = ['RANDOM','HUMAN']
Dict[10] = ['RANDOM','CONSOLE']
Dict[11] = ['MINMAX','MINMAX ALPHA-BETA PRUNING']
Dict[12] = ['MINMAX','MONTE']
Dict[13] = ['MINMAX','HUMAN']
Dict[14] = ['MINMAX','CONSOLE']
Dict[15] = ['MINMAX ALPHA-BETA PRUNING','MONTE']
Dict[16] = ['MINMAX ALPHA-BETA PRUNING','HUMAN']
Dict[17] = ['MINMAX ALPHA-BETA PRUNING','CONSOLE']
Dict[18] = ['MONTE','HUMAN']
Dict[19] = ['MONTE','CONSOLE']

def menu():
    title = 'Please choose your favorite programming language: '
    options = []
    options.append(' 1. BOTH are Random')
    options.append(' 2. BOTH are MinMax')
    options.append(' 3. BOTH are MinMax Alpha-Beta Pruning')
    options.append(' 4. BOTH are Monte Carlo Tree Search')
    options.append(' 5. BOTH are Human on App')
    options.append(' 6. BOTH are Human on Console')
    options.append(' 7. Random VS MinMax')
    options.append(' 8. Random VS MinMax Alpha-Beta Pruning')
    options.append(' 9. Random VS Monte Carlo Tree Search')
    options.append('10. Random VS Human on App')
    options.append('11. Random VS Human on Console')
    options.append('12. MinMax VS MinMax Alpha-Beta Pruning')
    options.append('13. MinMax VS Monte Carlo Tree Search')
    options.append('14. MinMax VS Human on App')
    options.append('15. MinMax VS Human on Console')
    options.append('16. MinMax Alpha-Beta Pruning VS Monte Carlo Tree Search')
    options.append('17. MinMax Alpha-Beta Pruning VS Human on App')
    options.append('18. MinMax Alpha-Beta Pruning VS Human on Console')
    options.append('19. Monte Carlo Tree Search VS Human on App')
    options.append('20. Monte Carlo Tree Search VS Human on Console')
    option = pick(options,indicator='=>')
    index_option = option[0][1]
    selection = Dict[index_option]
    P1 = selection[0]
    P2 = selection[1]
    print(selection)
    return P1, P2

#===========================================

class Owela_Game:
    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.currPlayer = 0
        self.never_ending = False
        self.state = [
            [2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 0, 0, 0, 0, 0]]
    
    def Begin(self):
        pygame.init()
        self._screen = pygame.display.set_mode((640,480),0,32)
        pygame.display.set_caption("OWELA")
        self._clock = pygame.time.Clock()
        self._basicFont = pygame.font.SysFont(None, 48)
        
        self._screen.fill(WHITE)
        self.redraw()
        pygame.display.update()
    
    def animate(self):
        self._screen.fill(WHITE)
        P1_text = self._basicFont.render(self.player_1+' PLAYER - STONE COUNT : '+str(self.stone_count(0)), True, RED,   WHITE)
        P2_text = self._basicFont.render(self.player_2+' PLAYER - STONE COUNT : '+str(self.stone_count(1)), True, GREEN, WHITE)
        
        P1_textRect = P1_text.get_rect()
        P1_textRect.centerx = 300
        P1_textRect.centery = 50
        
        P2_textRect = P2_text.get_rect()
        P2_textRect.centerx = 300
        P2_textRect.centery = 400
        
        self._screen.blit(P1_text,P1_textRect)
        self._screen.blit(P2_text,P2_textRect)
        
        self.redraw()
        pygame.display.update()
        pygame.time.delay(3000)
    
    def copy(self):
        copied_game = Owela_Game(self.player_1, self.player_2)
        copied_game.state[0] = self.state[0][:]
        copied_game.state[1] = self.state[1][:]
        copied_game.currPlayer = self.currPlayer
        return copied_game
    
    def swapPlayers(self):
        if self.currPlayer == 0:
            self.currPlayer = 1
        else:
            self.currPlayer = 0
    
    def getValidPlays(self):
        valid_moves = []
        for i in range(16):
            if self.state[self.currPlayer ][i] > 0:
                valid_moves.append(i)
        return valid_moves
    
    def stone_count(self, player):
        return sum(self.state[player])
    
    def player_has_won(self, player):
        return self.stone_count(1 - player) <= 1
    
    def gameDone(self):
        return self.player_has_won(0) or self.player_has_won(1)

    def get_winner(self):
        if self.player_has_won(0):
            return 0
        elif self.player_has_won(1):
            return 1
    
    def __repr__(self):
        return dedent(f"""\
            State: {list(reversed(self.state[0][:8]))}
                   {self.state[0][8:]}
                   ------------------------
                   {list(reversed(self.state[1][8:]))}
                   {self.state[1][:8]}\
        """)
    
    def compute_final_reward(self, player):
        return self.stone_count(player) - self.stone_count(1 - player)
    
    def compute_end_game_reward(self, player):
        val = None
        if self.player_2 == 'MONTE':
            val = self.stone_count(1) - self.stone_count(0)
        elif self.player_1 == 'MONTE':
            val = self.stone_count(0) - self.stone_count(1)
        return val
        
    def has_direct_winning_move(self):
        moves_possible = self.getValidPlays()
        for pos in moves_possible:
            game_copy = self.copy()
            game_copy.make_move(pos)
            if game_copy.player_has_won(self.currPlayer):
                return pos
        return None
    
    def make_move(self, position):
        total_stolen, keep_old_stones, capture, count = 0, -1, None, 0
        player = self.currPlayer
        if self.state[player][position] <= 0:
            raise Exception("invalid", position)
        my_state = self.state[player]
        other_state = self.state[1 - player]
        while True:
            count = count + 1
            if self.player_has_won(player):
                break
            move_ = str(self.state)
            if capture is None:
                capture = move_
            else:
                if (move_ == capture) or (count > 1000):
                    self.never_ending = True
                    break
            if keep_old_stones != -1:
                amount = (my_state[position])-keep_old_stones
                my_state[position] = keep_old_stones
                keep_old_stones = -1
            else:
                amount = my_state[position]
                my_state[position] = 0
            
            for i in range(1, amount + 1):
                my_state[(position + i) % 16] += 1
            new_position = (position + amount) % 16
            old_stones = int(my_state[new_position])
            
            if my_state[new_position] > 1:
                if  new_position >= 8:
                    steal_position_1 = new_position - 8
                    steal_position_2 = 15 - steal_position_1
                    val_at_front = other_state[steal_position_2]
                    
                    if val_at_front > 0:
                        stolen = other_state[steal_position_1] + other_state[steal_position_2]
                        other_state[steal_position_1] = 0
                        other_state[steal_position_2] = 0
                        my_state[new_position] += stolen
                        total_stolen += stolen
                        keep_old_stones = old_stones
                position = new_position
            else:
                break
        self.swapPlayers()
        return total_stolen
    
    def redraw(self):
        self._basicFont = pygame.font.SysFont(None, 25)
        board = pygame.Rect(0,70,640,280)#left, top, width, height #BROWN BOARD
        pygame.draw.rect(self._screen,BROWN,board)
        visual_state = [ list(reversed(self.state[0][:8])), self.state[0][8:] , list(reversed(self.state[1][8:])) , self.state[1][:8] ]
        #-------------------
        row_val = 120
        self.all_storage = []# stores rects for clicking on circles
        for row in range(4):
            col_val = 110
            for cnt in range(self.board_length):
                gen_position = (col_val, row_val)
                #Draw the pits
                if (self.currPlayer == 1) and (row in [2,3]):
                    pygame.draw.circle(self._screen,PINK,gen_position,28,2)
                elif (self.currPlayer == 0) and (row in [0,1]):
                    pygame.draw.circle(self._screen,PINK,gen_position,28,2)
                else:
                    pygame.draw.circle(self._screen,BLACK,gen_position,28,1)
                col_val = col_val + 65
                #------------
                m = pygame.Rect(0,0,30,30)
                m.center = gen_position
                stone_count = visual_state[row][cnt]
                for k in range(stone_count):
                    x = random.randint(m.x,m.x+m.w)
                    y = random.randint(m.y,m.y+m.h)
                    #Enter stones in pit
                    stone_img = pygame.image.load('stone.png')
                    stone_img = pygame.transform.scale(stone_img, (15, 15))
                    self._screen.blit(stone_img, (x-15,y-15))
                #Write count of stones, and save storage for user input
                text_created = self._basicFont.render(str(stone_count), True, BLACK,BROWN)
                self._screen.blit(text_created,m.center)
                if stone_count > 0:
                    count_new = cnt
                    if row == 0:
                        count_new = (cnt - 7)*-1
                    if row == 1:
                        count_new = cnt + 8
                    if row == 2:
                        count_new = 8 + (7-cnt)
                    self.all_storage.append([m,row,count_new])
            row_val = row_val + 60

#===========================================

class Node(object):
    def __init__(self, game: Owela_Game, move = None, parent=None):
        self.visits = 0
        self.reward = 0
        self.game = game
        self.children = []
        self.parent = parent
        self.move = move
        self.value = -1
        self.unexplored_moves = set(game.getValidPlays())

    @staticmethod
    def clone(other_node):
        return deepcopy(other_node)

    def put_child(self, child):
        self.children.append(child)
        self.unexplored_moves.remove(child.move)

    def update(self, reward):
        self.reward += reward
        self.visits += 1
        for child in self.children:
            self.value = max(self.value, child.value)

    def is_fully_expanded(self) -> bool:
        """ is_fully_expanded returns true if there are no more moves to explore. """
        return len(self.unexplored_moves) == 0

    def is_terminal(self) -> bool:
        """is_terminal returns true if the node is leaf node"""
        return len(self.game.getValidPlays()) == 0

    def __str__(self):
        return "Node; Move %s, number of children: %d; visits: %d; reward: %f; value: %f" % (
            self.move, len(self.children), self.visits, self.reward, self.value)

#===========================================

class Player():
    def __init__(self, player_number, player_type, owela, maximum_depth = float("inf")):
        self.player_number = player_number
        self.player_type = player_type
        self.opposite_player_number = self.get_opposite_player(self.player_number)
        self.owela = owela
        self.maximum_depth = maximum_depth

    def get_opposite_player(self, current_player):
        return 2 if current_player == 1 else 1

    def calculate_winner(self, game):
        player_one_score = game.stone_count(0)
        player_two_score = game.stone_count(1)
        if player_one_score > player_two_score:
            return 1
        elif player_one_score < player_two_score:
            return 2
        else:
            return 0

    def minimax_score(self, game_instance, depth, game_over):
        winner = None
        if game_over:
            winner = (game_instance.get_winner()) + 1
        else:
            winner = self.calculate_winner(game_instance)
        #----------
        if winner is self.player_number:
            return 100 - depth
        elif winner is self.opposite_player_number:
            return 0 - depth
        else:
            return 50

    def minimax(self, game, phasing_player, depth = 0, move = None):
        game_over = game.gameDone()
        if game_over or depth is self.maximum_depth:
            return [self.minimax_score(game, depth, game_over), move]

        depth += 1
        scores = []
        moves = []
        legal_moves = game.getValidPlays()
        next_player = self.get_opposite_player(phasing_player)

        for move in legal_moves:
            possible_game = game.copy()
            possible_game.make_move(move)
            if possible_game.never_ending != True:
                score = self.minimax(possible_game, next_player, depth, move)[0]
                scores.append(score)
                moves.append(move)

        # Do the min or max calculation
        if phasing_player == self.player_number:
            score_index = scores.index(max(scores))
            return [max(scores), moves[int(score_index)]]
        else:
            score_index = scores.index(min(scores))
            return [min(scores), moves[int(score_index)]]
    
    def minimax_alpha_beta_score(self, game_instance, depth, game_over):
        winner = None
        if game_over:
            winner = (game_instance.get_winner()) + 1
        else:
            winner = self.calculate_winner(game_instance)
        #----------
        if winner is self.player_number or winner is self.opposite_player_number:
            val = None
            if game_instance.player_2 == 'MINMAX ALPHA-BETA PRUNING':
                val = game_instance.stone_count(1) - game_instance.stone_count(0)
            elif game_instance.player_1 == 'MINMAX ALPHA-BETA PRUNING':
                val = game_instance.stone_count(0) - game_instance.stone_count(1)
            return val
        else:
            return 0

    def minimax_alpha_beta(self, game, phasing_player, alpha = float("-inf"), beta = float("inf"), best_move = None, depth = -1):
        game_over = game.gameDone()
        if game_over:
            return [self.minimax_alpha_beta_score(game, depth, game_over), best_move]
        elif depth is self.maximum_depth:
            return [self.minimax_alpha_beta_score(game, depth, game_over), best_move]

        depth += 1
        legal_moves = game.getValidPlays()
        next_player = self.get_opposite_player(phasing_player)

        for move in legal_moves:
            possible_game = game.copy()
            possible_game.make_move(move)
            if phasing_player == self.player_number: #is maximising node
                result = self.minimax_alpha_beta(possible_game, next_player, alpha, beta, move, depth)[0]
                if result > alpha:
                    alpha = result
                    best_move = move
                if alpha >= beta: # Pruning
                    break
            
            else: #is minimising node
                result = self.minimax_alpha_beta(possible_game, next_player, alpha, beta, move, depth)[0]
                if result < beta:
                    beta = result
                    best_move = move
                if beta <= alpha:  # Pruning
                    break

        best_score = alpha if phasing_player == self.player_number else beta
        return [best_score, best_move]

    def pick_minimax_alpha_beta(self):
        return self.minimax_alpha_beta(self.owela, self.player_number)[1]

    def pick_minimax(self):
        return self.minimax(self.owela, self.player_number)[1]
    
    def backpropagate(self, root: Node, final_state: Owela_Game):#RollOutPolicy -->has the strategy for propagating rewards up the tree
        """
        backpropgate pushes the reward (pay/visits) to the parents node up to the root
        :param root: starting node to backpropgate from
        :final_state: the state of final node (holds final reward from the simulation)
        """
        node = root
        # propagate node reward to parents'
        while node is not None:
            reward = final_state.compute_end_game_reward(self.player_number-1)
            node.update(reward)
            node = node.parent
    
    def simulate(self, root: Node) -> Owela_Game:#MonteCarloDefaultPolicy plays the domain randomly from a given non-terminal state.
        node = Node.clone(root)
        while not node.game.gameDone():
            legal_moves = node.game.getValidPlays()
            moves = []
            move_to_make = None
            high_score = 0
            for move in legal_moves:
                game_copy = node.game.copy()
                score = game_copy.make_move(move)
                if game_copy.never_ending != True:
                    if score > high_score:
                        move_to_make = move
                        high_score = score
            if move_to_make == None:
                move_to_make = random.choice(legal_moves)
            node.game.make_move(move_to_make)
        return node.game
     
    def get_max_child(self, node: Node) -> Node:
        """select_max_child returns the child with highest average reward."""
        if node.is_terminal():
            raise ValueError('Terminal node; there are no children to select from.')
        if len(node.children) == 0:
            raise ValueError('Selecting max child from unexpanded node')
        elif len(node.children) == 1:
            return node.children[0]
        return max(node.children, key=lambda child: child.reward / child.visits)
        
    def rave_selection(self, node: Node) -> Node:
        """returns the child that maximise the heuristic value."""
        if node.is_terminal() and node.is_fully_expanded():
            raise ValueError('Terminal node; there are no children to select from.')
        elif len(node.children) == 1:
            return node.children[0]
        return max(node.children, key=lambda child: self._uct_rave_reward(node, child))
    
    def _uct_rave_reward(self, root: Node, child: Node, exploration_constant: float = 0.5) -> float:
        return self._rave_reward(child) + (exploration_constant * sqrt(log(root.visits) / child.visits))
    
    def _rave_reward(self, node: Node, alpha: float = 0.5) -> float:
        return (1 - alpha) * (node.reward / node.visits) + alpha * node.value
    
    """TreePolicy selects and expands from the nodes already contained within the search tree."""
    def monte_select(self, node: Node) -> Node:
        while not node.is_terminal():
            # expand while we have nodes to expand
            if not node.is_fully_expanded():
                return self.expand(node)
            # select child and explore it
            else:
                node = self.rave_selection(node)
        return node
    
    def expand(self, parent: Node) -> Node:
        child_node = None
        for child_expansion_move in tuple(parent.unexplored_moves):
            child_state = parent.game.copy()
            child_state.make_move(child_expansion_move)
            child_node_ = Node(game=child_state, move=child_expansion_move, parent=parent)
            if child_node_.game.gameDone() == False:
                child_node = child_node_
                break
        if child_node != None:
            parent.put_child(child_node)
            self._rave_expand(child_node)
            # go down the tree
            return child_node
        else:
            return self.rave_selection(parent)
    
    def _rave_expand(self, parent: Node):
        scores = []
        legal_moves = parent.unexplored_moves
        for unexplored_move in legal_moves:
            child_state = parent.game.copy()#??
            child_state.make_move(unexplored_move)#??
            scores.append( (parent.game.copy()).make_move(unexplored_move) )

        moves_dist = np.asarray(scores, dtype=np.float64).flatten()
        exp = np.exp(moves_dist - np.max(moves_dist))
        dist = exp / np.sum(exp)
        parent.value = max(dist)
    
    def monte_carlo(self, time_sec: int):
        game = self.owela
        calculation_time: datetime.timedelta = datetime.timedelta(seconds=time_sec)
        
        game_state_root = Node(game=game.copy())
        start_time = datetime.datetime.utcnow()
        games_played = 0
        while datetime.datetime.utcnow() - start_time < calculation_time:
            node = self.monte_select(game_state_root)
            final_state = self.simulate(node)
            self.backpropagate(node, final_state)
            games_played += 1
            if monte_log == True:
                print("%s; Game played %i" % (node, games_played))
        chosen_child = self.get_max_child(game_state_root)
        if monte_log == True:
            print("%s" % game_state_root)
            print("Choosing: %s" % chosen_child)
        return chosen_child.move

    def select_move(self):
        game_ = self.owela
        # short circuit last move
        if len(game_.getValidPlays()) == 1:
            return game_.getValidPlays()[0]
        #----------        
        pot = None
        if self.player_type == "RANDOM":
            pot = random.choice(self.owela.getValidPlays())
            return pot
        #----------
        pot = game_.has_direct_winning_move()
        if pot is not None:
            return pot
        #----------
        if self.player_type == "MINMAX":
            pot = self.pick_minimax()

        if self.player_type == "MINMAX ALPHA-BETA PRUNING":
            pot = self.pick_minimax_alpha_beta()
        
        if self.player_type == "MONTE":
            pot = self.monte_carlo(1)
        return pot

#===========================================

def get_human_move(game):
    move = None
    while move is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i in game.all_storage:
                    if game.currPlayer == 0:
                        if i[1] in [0,1]:
                            click_pos = i[0]
                            click_move = i[2]
                            if click_pos.collidepoint(pygame.mouse.get_pos()):
                                move = click_move
                    elif game.currPlayer == 1:
                        if i[1] in [2,3]:
                            click_pos = i[0]
                            click_move = i[2]
                            if click_pos.collidepoint(pygame.mouse.get_pos()):
                                move = click_move
    return move

#===========================================

def get_console_move(game, p):
    options = game.getValidPlays()
    move = input(p+" SELECT MOVE FROM "+str(options)+" :")
    move = int(move)
    while move not in options:
        print('INVALID OPTION ENTERED')
        move = input(p+" SELECT MOVE FROM "+str(options)+" :")
        move = int(move)
    return move

#===========================================

def play_game(P1,P2):
    minmax_time = 0
    alpha_beta_time = 0
    game = Owela_Game(P1,P2)
    if 'HUMAN' in [P1,P2]:
        game.Begin()
    if (print_log == True) or ('CONSOLE' in [P1,P2]):
        print(game)
    while not game.gameDone():
        if (print_log == True) or ('CONSOLE' in [P1,P2]):
            print('======================================================')
        if game.currPlayer == 0:
            move = None
            if P1 == 'HUMAN':
                move = get_human_move(game)
            elif P1 == 'CONSOLE':
                move = get_console_move(game,'P1')
            else:
                time_start = time.perf_counter()
                move = Player(1, P1, game, max_depth).select_move()
                time_end = time.perf_counter()
                time_duration = time_end - time_start
                if P1 == 'MINMAX':
                    minmax_time += time_duration
                if P1 == 'MINMAX ALPHA-BETA PRUNING':
                    alpha_beta_time += time_duration
            if (print_log == True) or ('CONSOLE' in [P1,P2]):
                print('P1 - ',P1, 'MOVE : ', move)
            game.make_move(move)
            if 'HUMAN' in [P1,P2]:
                game.animate()
            if game.never_ending == True:
                return ['','DRAW',minmax_time,alpha_beta_time]
            if game.gameDone():
                if 'HUMAN' in [P1,P2]:
                    pygame.quit()
                if (print_log == True) or ('CONSOLE' in [P1,P2]):
                    print(game)
                return ['P1 - ',P1,minmax_time,alpha_beta_time]
        else:
            move = None
            if P2 == 'HUMAN':
                move = get_human_move(game)
            elif P2 == 'CONSOLE':
                move = get_console_move(game,'P2')
            else:
                time_start = time.perf_counter()
                move = Player(2, P2, game, max_depth).select_move()
                time_end = time.perf_counter()
                time_duration = time_end - time_start
                if P2 == 'MINMAX':
                    minmax_time += time_duration
                if P2 == 'MINMAX ALPHA-BETA PRUNING':
                    alpha_beta_time += time_duration
            if (print_log == True) or ('CONSOLE' in [P1,P2]):
                print('P2 - ',P2,' MOVE : ', move)
            game.make_move(move)
            if 'HUMAN' in [P1,P2]:
                game.animate()
            if game.never_ending == True:
                return ['','DRAW',minmax_time,alpha_beta_time]
            if game.gameDone():
                if 'HUMAN' in [P1,P2]:
                    pygame.quit()
                if (print_log == True) or ('CONSOLE' in [P1,P2]):
                    print(game)
                return ['P2 - ' ,P2,minmax_time,alpha_beta_time]
        if (print_log == True) or ('CONSOLE' in [P1,P2]):
            print(game)
            print(game.state)
            print('SCORE : (P1 -',game.stone_count(0),") : (P2 -",game.stone_count(1),')')

#==================================================
if __name__ == '__main__':
    P1,P2 = menu()
    if ('CONSOLE' in [P1,P2]) or ('HUMAN' in [P1,P2]) or (test == True):
        tournaments = 1
        number_of_games = 1
    for tournament in range(tournaments):
        results = []
        for number in range(number_of_games):
            result = play_game(P1,P2)
            winner = result[0] + result[1]
            if result[1] == 'MINMAX':
                results.append([number, winner, result[2]])
                print('TOURNAMENT : ',(tournament+1), 'GAME :',(number+1),' WINNER : ', winner, ' TIME : ', result[2])
            elif result[1] == 'MINMAX ALPHA-BETA PRUNING':
                results.append([number, winner, result[3]])
                print('TOURNAMENT : ',(tournament+1), 'GAME :',(number+1),' WINNER : ', winner, ' TIME : ', result[3])
            else:
                results.append([number, winner, 0])
                print('TOURNAMENT : ',(tournament+1), 'GAME :',(number+1),' WINNER : ', winner)
        df = pd.DataFrame(results, columns=["GAME_NO", "WINNER", "SECONDS"])
        df_summary = ps.sqldf("SELECT WINNER,count(*) as NUMBER_of_GAMES_WON, sum(SECONDS) TOTAL_SECONDS FROM df group by WINNER", locals())
        print('_________________________________________________')
        print(df_summary)

#==================================================