import os
import time
from typing import Optional
from anthropic import Anthropic
import numpy as np
from game import Move, move, print_board, init_board, IllegalMove, get_legal_moves


class GameAgent:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the 2048 game agent with Anthropic API."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable must be set or passed as parameter"
            )

        self.client = Anthropic(api_key=self.api_key)
        self.board = None
        self.move_count = 0

    def reset_game(self):
        """Reset the game to initial state."""
        self.board = init_board()
        self.move_count = 0
        return self.board

    def board_to_string(self, board: np.ndarray) -> str:
        """Convert board to a readable string format."""
        board_str = ""
        for i in range(4):
            row = []
            for j in range(4):
                val = board[i, j]
                row.append(str(val) if val > 0 else ".")
            board_str += " ".join(f"{cell:>4}" for cell in row) + "\n"
        return board_str

    def get_next_move(self) -> tuple[Optional[Move], str]:
        """Get the next move from the AI agent along with reasoning."""
        if self.board is None:
            return None, ""

        available_moves = get_legal_moves(self.board)
        if not available_moves:
            return None, "Game over - no moves available"  # Game over

        post_move_state_str = ""
        for available_move in available_moves:
            new_board = move(self.board, available_move, add_tile=False)
            post_move_state_str += f"Board after {available_move.name} move:\n{self.board_to_string(new_board)}\n"

        # Prepare the prompt for the AI
        board_str = self.board_to_string(self.board)

        prompt = f"""You are playing the game 2048. Your goal is to merge tiles to reach the 2048 tile and achieve the highest possible highest tile.
Valid moves are ones that result in at least one tile changing its position, or at least one merge occurring between two tiles of the same value.
After a valid move, a new tile will be randomly generated in an empty cell. 90% of the time the new tile will be a 2, 10% of the time it will be a 4.

Current board state (move #{self.move_count + 1}, highest tile: {np.max(self.board)}):
{board_str}

Here are the possible board states after each move, before the random new tile is added:
{post_move_state_str}

Available moves: {[move.name for move in available_moves]}

First explain your reasoning, then provide your move choice in the format MOVE: [LEFT/RIGHT/UP/DOWN].
Do not say anything after the move choice, or I will not be able to parse your selected move.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        print(len(response.content))
        print(response.content[0].text)

        # Extract reasoning and move
        reasoning = ""
        move_text = ""

        lines = response_text.split("\n")
        # Pop last line as it is the move
        move_text = lines.pop()

        # Selected move text is of the form "MOVE: [LEFT/RIGHT/UP/DOWN]"
        try:
            selected_move = move_text.split(": ")[1]
        except IndexError:
            print(move_text)
            return
        if selected_move == "LEFT":
            selected_move = Move.LEFT
        elif selected_move == "RIGHT":
            selected_move = Move.RIGHT
        elif selected_move == "UP":
            selected_move = Move.UP
        elif selected_move == "DOWN":
            selected_move = Move.DOWN

        reasoning_lines = []
        for line in lines:
            reasoning_lines.append(line)
        reasoning = "\n".join(reasoning_lines).strip()

        return selected_move, reasoning


    def make_move(self, move_direction: Move) -> bool:
        """Make a move and update the game state."""
        if self.board is None:
            return False

        try:
            self.board = move(self.board, move_direction)
            self.move_count += 1
            return True
        except IllegalMove:
            return False

    def play_game(
        self, max_moves: int = 1000, show_board: bool = True
    ):
        """Play a complete game with the AI agent."""
        self.reset_game()

        if show_board:
            print("Starting new 2048 game with AI agent...")
            print_board(self.board)
            print("=" * 50)

        while self.move_count < max_moves:
            next_move, reasoning = self.get_next_move()
            if next_move is None:
                if show_board:
                    print("Game over! No more moves available.")
                break

            if show_board:
                print(f"AI chooses: {next_move}")
                # print(f"Reasoning: {reasoning}")

            success = self.make_move(next_move)
            if not success:
                if show_board:
                    print("Invalid move attempted!")
                break

            if show_board:
                print("-" * 50)
                print_board(self.board)
                print(f"Move #{self.move_count}, Highest tile: {np.max(self.board)}")
                print("=" * 50)

        if show_board:
            print(f"Game finished after {self.move_count} moves!")
            print(f"Highest tile: {np.max(self.board)}")

        return {
            "moves": self.move_count,
            "highest_tile": int(np.max(self.board)),
            "final_board": self.board.copy(),
        }


def main():
    """Main function to run the AI agent."""
    agent = GameAgent()

    # Play a single game
    result = agent.play_game(max_moves=1000, show_board=True)

    print("\n" + "=" * 50)
    print("GAME SUMMARY")
    print("=" * 50)
    print(f"Total moves: {result['moves']}")
    print(f"Highest tile achieved: {result['highest_tile']}")

if __name__ == "__main__":
    main()
