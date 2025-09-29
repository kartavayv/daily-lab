class tic_tac_toe:
    def __init__(self):
        self.board = [" "  for _ in range(9)]
        self.player = "O"
        self.ai_player = "X"
    
    def get_board(self):
        for i in range(0,9,3):
            print(f"{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i< 6: #heart!
                print("-------------")
    def available_moves(self):
        return [ind for ind, spot in enumerate(self.board) if spot == " "]
    
    def make_move(self,position, player):
        if self.board[position] == " ":
            self.board[position] = player
            return True
        return False
    
    def is_board_full(self):
        return " " not in self.board
    
    def check_winner(self):
        #rows
        for i in range(0,9,3):
            if self.board[i]  == self.board[i+1] == self.board[i+2] != " ":
                return self.board[i]
        
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] != " ":
                return self.board[i]
        
        if self.board[0] == self.board[4] == self.board[8] != " ":
            return self.board[0]
        
        if self.board[2] == self.board[4] == self.board[6] != " ":
            return self.board[2]
        
        return None
    
    def game_over(self):
        return self.check_winner() is not None or self.is_board_full()
    
    def minimax(self, depth, is_maximizing):
        if self.check_winner() == self.ai_player:
            return 1
        
        if self.check_winner() == self.player:
            return -1
        
        if self.is_board_full():
            return 0
        
        if is_maximizing:
            best_score = float("-inf") #records the max value of all the possible available moves and since it's recursive so this does it for different level of possibility tree
            for move in self.available_moves():
                self.board[move] = self.ai_player
                score = self.minimax(depth+1, False) #player's turn not ai's
                best_score = max(score, best_score) #always passes the max value above out of all the possibilities
                self.board[move] = " "

            return best_score
        else:
            best_score = float("inf") #records the min value of all the possible available moves and since it's recursive so this does it for different level of possibility tree
            for move in self.available_moves():
                self.board[move] = self.player
                score = self.minimax(depth+1, True)
                self.board[move] = " "
                best_score = min(score, best_score) #always passes the min value above out of all the possibilities
            
            return best_score

    def get_best_move(self):
        best_score = float("-inf")
        best_move = None

        for move in self.available_moves():
            self.board[move] = self.ai_player
            score = self.minimax(0, False)

            if score > best_score:
                best_score = score
                best_move = move
            
            self.board[move] = " "

        return best_move
    
    def play_game(self):
        print("WELCOME TO THE TIC TAC TOE GAME. GET READY TO BE CRUSHED BY AI!")
        print("YOU ARE 'O' AND AI IS 'X' ")
        print("ENTER THE MOVES YOU WANNA PLAY BELOW")


        while True:

            if self.game_over():
                if self.is_board_full():
                    print("THE GAME IS A TIE")
                else:
                    print("THE WINNER IS: ", self.check_winner())
                
                break

            player_move = int(input("ENTER YOUR MOVE: ")) #let the player think as a player, first block is 1 not 0, so using -1 below

            if (player_move-1) not in self.available_moves():
                print("ENTER A VALID MOVE")
                player_move = int(input("ENTER YOUR MOVE: "))
                
            self.board[player_move-1] = "O"
            print("Your move...")
            self.get_board()
            print("\n")
            ai_move = int(self.get_best_move())
            self.board[ai_move] = "X"
            print("AI's move...")
            self.get_board()
            print("\n")

    # ... the rest of the class
    # def play_game(self):
                
    #     print("WELCOME TO THE TIC TAC TOE GAME. GET READY TO BE CRUSHED BY AI!")
    #     print("YOU ARE 'O' AND AI IS 'X' ")
    #     print("ENTER THE MOVES YOU WANNA PLAY BELOW")

    # # Randomly decide who goes first
    #     import random

    #     ai_turn = random.choice([True, False])

    #     while not self.game_over():
    #         self.get_board()

    #         if ai_turn:
    #             print("\nAI's turn...")
    #             move = self.get_best_move()
    #             self.make_move(move, self.ai_player)
    #         else:
    #             while True:
    #                 try:
    #                     move = int(input("\nYour turn (0-8): "))
    #                     if 0 <= move <= 8 and self.make_move(move, self.player):
    #                         break
    #                     else:
    #                         print("Invalid move! Try again.")
    #                 except ValueError:
    #                     print("Please enter a number between 0 and 8!")

    #         ai_turn = not ai_turn





if __name__ == "__main__":
    game = tic_tac_toe()
    game.play_game()

# game.make_move(0, "X")
# game.make_move(4, "X")
# game.make_move(8, "X")
# game.get_board()
# print(game.available_moves())
# print(game.check_winner())
# print("the best move:", game.get_best_move())