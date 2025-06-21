import os
import time
from typing import Optional
from anthropic import Anthropic
import numpy as np
from game import Move, move, print_board, init_board, IllegalMove


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
        self.score = 0

    def reset_game(self):
        """Reset the game to initial state."""
        self.board = init_board()
        self.move_count = 0
        self.score = 0
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

    def calculate_score(self, board: np.ndarray) -> int:
        """Calculate the current score based on tile values."""
        return int(np.sum(board))

    def get_available_moves(self, board: np.ndarray) -> list[Move]:
        """Get all available legal moves for the current board state."""
        available_moves = []
        for move_direction in Move:
            try:
                # Try the move to see if it's legal
                test_board = board.copy()
                move(test_board, move_direction)
                available_moves.append(move_direction)
            except IllegalMove:
                continue
        return available_moves

    def get_next_move(self) -> tuple[Optional[Move], str]:
        """Get the next move from the AI agent along with reasoning."""
        if self.board is None:
            return None, ""

        available_moves = self.get_available_moves(self.board)
        if not available_moves:
            return None, "Game over - no moves available"  # Game over

        # Prepare the prompt for the AI
        board_str = self.board_to_string(self.board)
        current_score = self.calculate_score(self.board)

        prompt = f"""You are playing the game 2048. Your goal is to merge tiles to reach the 2048 tile and achieve the highest possible score.

Current board state (move #{self.move_count + 1}, score: {current_score}):
{board_str}

Available moves: {[move.name for move in available_moves]}

In 2048, the best strategy typically involves:
1. Keeping the highest value tile in a corner (usually bottom-right)
2. Building a snake-like pattern from that corner
3. Avoiding random moves that break the pattern
4. Prioritizing moves that merge tiles and create space

Please analyze the current board and choose the best move from the available options. Consider:
- Which move will create the most merges?
- Which move maintains or improves the board structure?
- Which move opens up the most opportunities for future moves?
When considering each move, on a new line print your expectation for what the board will look like after the move, excluding the new randomly generated tile after the move.


First explain your reasoning, then provide your move choice.

Format your response as:
<For each candidate move>
CANDIDATE MOVE: [LEFT/RIGHT/UP/DOWN]
BOARD AFTER MOVE: [board after move]
REASONING: [your analysis here]
<Finally>
MOVE: [LEFT/RIGHT/UP/DOWN]"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Extract reasoning and move
            reasoning = ""
            move_text = ""

            # First try to find the move in the response
            for move_name in ["LEFT", "RIGHT", "UP", "DOWN"]:
                if move_name in response_text:
                    move_text = move_name
                    break

            # If we found a move, extract everything except the move line as reasoning
            if move_text:
                lines = response_text.split("\n")
                reasoning_lines = []
                for line in lines:
                    if move_text not in line.strip():
                        reasoning_lines.append(line)
                reasoning = "\n".join(reasoning_lines).strip()
            else:
                # If no move found, use entire response as reasoning
                reasoning = response_text.strip()

            # Parse the move
            selected_move = None
            for move_direction in available_moves:
                if move_direction.name in move_text:
                    selected_move = move_direction
                    break

            # If no exact match, try to infer from the response
            if selected_move is None:
                if "LEFT" in move_text and Move.LEFT in available_moves:
                    selected_move = Move.LEFT
                elif "RIGHT" in move_text and Move.RIGHT in available_moves:
                    selected_move = Move.RIGHT
                elif "UP" in move_text and Move.UP in available_moves:
                    selected_move = Move.UP
                elif "DOWN" in move_text and Move.DOWN in available_moves:
                    selected_move = Move.DOWN
                else:
                    # Fallback to first available move
                    selected_move = available_moves[0]
                    reasoning += " (Fallback: no clear move found in response)"

            return selected_move, reasoning

        except Exception as e:
            print(f"Error getting AI move: {e}")
            # Fallback to first available move
            fallback_move = available_moves[0] if available_moves else None
            return fallback_move, f"Error occurred: {e}"

    def make_move(self, move_direction: Move) -> bool:
        """Make a move and update the game state."""
        if self.board is None:
            return False

        try:
            self.board = move(self.board, move_direction)
            self.move_count += 1
            self.score = self.calculate_score(self.board)
            return True
        except IllegalMove:
            return False

    def play_game(
        self, max_moves: int = 1000, delay: float = 0.5, show_board: bool = True
    ):
        """Play a complete game with the AI agent."""
        self.reset_game()

        if show_board:
            print("Starting new 2048 game with AI agent...")
            print_board(self.board)
            print(f"Score: {self.score}")
            print("-" * 50)

        while self.move_count < max_moves:
            next_move, reasoning = self.get_next_move()
            if next_move is None:
                if show_board:
                    print("Game over! No more moves available.")
                break

            if show_board:
                print(f"AI chooses: {next_move.name}")
                print(f"Reasoning: {reasoning}")

            success = self.make_move(next_move)
            if not success:
                if show_board:
                    print("Invalid move attempted!")
                break

            if show_board:
                print_board(self.board)
                print(f"Move #{self.move_count}, Score: {self.score}")
                print("-" * 50)
                time.sleep(delay)

        if show_board:
            print(f"Game finished after {self.move_count} moves!")
            print(f"Final score: {self.score}")
            print(f"Highest tile: {np.max(self.board)}")

        return {
            "moves": self.move_count,
            "score": self.score,
            "highest_tile": int(np.max(self.board)),
            "final_board": self.board.copy(),
        }


def main():
    """Main function to run the AI agent."""
    try:
        agent = GameAgent()

        # Play a single game
        result = agent.play_game(max_moves=1000, delay=0.5, show_board=True)

        print("\n" + "=" * 50)
        print("GAME SUMMARY")
        print("=" * 50)
        print(f"Total moves: {result['moves']}")
        print(f"Final score: {result['score']}")
        print(f"Highest tile achieved: {result['highest_tile']}")

    except ValueError as e:
        print(f"Error: {e}")
        print(
            "Please set the ANTHROPIC_API_KEY environment variable or pass it as a parameter."
        )
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
