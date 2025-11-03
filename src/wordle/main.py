import importlib.resources
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from wordle.solver import (get_word, load_words, pick_best_word,
                           process_result, words_to_arr)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # RichHandler will format the rest
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)

app = typer.Typer(help="A command-line Wordle solver.")
console = Console()


def get_guess(suggested_word: str, n_chars: int = 5) -> Optional[str]:
    while True:
        guess = (
            Prompt.ask(
                "Enter your guess (or 'q' to quit)",
                default=suggested_word,
            )
            .strip()
            .upper()
        )

        # Validate guess
        if guess == "Q":
            console.print("Exiting game.", style="italic")
            return None
        if len(guess) != n_chars:
            console.print(
                f"Error: Guess must be {n_chars} letters long. Try again.", style="bold"
            )
            continue
        if not guess.isalpha():
            console.print(
                "Error: Guess must only contain letters. Try again.", style="bold"
            )
            continue

        return guess


def get_response(n_chars: int = 5) -> str | None:
    # Create a styled prompt
    prompt_text = Text()
    prompt_text.append("\nEnter the response:\n")
    prompt_text.append(" (")
    prompt_text.append("G", style="bold")
    prompt_text.append(" = Green, ")
    prompt_text.append("Y", style="bold")
    prompt_text.append(" = Yellow, ")
    prompt_text.append("B", style="bold")
    prompt_text.append(" = Black)\n")
    prompt_text.append("Response (or 'q' to quit)")

    while True:
        resp = Prompt.ask(prompt_text).strip().upper()

        # Validate response
        if resp == "Q":
            console.print("Exiting game.", style="italic")
            return None
        if len(resp) != n_chars:
            console.print(
                f"Error: Response must be {n_chars} letters long. Try again.",
                style="bold",
            )
            continue
        if not set(resp).issubset({"G", "Y", "B"}):
            console.print(
                "Error: Response must only contain G, Y, or B. Try again.", style="bold"
            )
            continue

        return resp


def get_words_list(words_file: Optional[Path] = None, n_chars: int = 5) -> list[str]:
    if words_file is None:
        console.print("Using default word list.")
        try:
            # Use importlib.resources to get a path to the file
            default_words_file = importlib.resources.files("wordle").joinpath(
                "words.txt"
            )
            with importlib.resources.as_file(default_words_file) as words_file_path:
                return load_words(words_file_path, n_chars=n_chars)
        except FileNotFoundError:
            console.print(
                "Error: Default 'words.txt' not found within the package.", style="bold"
            )
            raise typer.Exit(code=1)
    else:
        console.print(f"Using custom word list from: [dim]{words_file}[/dim]")
        return load_words(words_file, n_chars=n_chars)


@app.command()
def main(
    words_file: Path = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a custom word list file. If not provided, uses a default list.",
    ),
):
    """
    Runs the command-line Wordle solver.
    """
    words_list = get_words_list(words_file, n_chars=5)
    if not words_list:
        console.print("No words loaded. Exiting.", style="bold")
        raise typer.Exit(code=1)

    words_arr = words_to_arr(words_list)
    n_chars = words_arr.shape[1]
    GREEN_WIN = "G" * n_chars

    console.print(
        f"Starting Wordle Solver CLI (Loaded {len(words_list)} words)", style="bold"
    )
    while True:
        n_remaining = len(words_arr)
        if n_remaining == 0:
            console.print("\n[bold]No possible words match the given clues.[/bold]")
            break

        console.print(f"\n[bold]{n_remaining}[/bold] possible words remaining")

        # Display the remaining words
        max_words = 128
        display_words = min(n_remaining, max_words)
        decoded_words = [get_word(words_arr, i) for i in range(display_words)]

        words_text = Text(", ").join(Text(w, style="dim") for w in decoded_words)
        if n_remaining > max_words:
            words_text.append(
                f" and {n_remaining - max_words} more...", style="italic dim"
            )
        console.print(words_text)

        # Suggest a word
        best_idx = pick_best_word(words_arr)
        if best_idx == -1:
            console.print("[bold]Error: Could not pick a best word.[/bold]")
            break
        suggested_word = get_word(words_arr, best_idx)
        console.print(
            Panel(
                f"[bold]{suggested_word}[/bold]",
                title="Suggested Guess",
                border_style="dim",
            )
        )

        # Let the user fill in the word and response
        guess = get_guess(suggested_word, n_chars)
        if guess is None:
            return
        resp = get_response(n_chars)
        if resp is None:
            return

        # If the response is GGGGG print congrats and exit ---
        if resp == GREEN_WIN:
            console.print("\n[bold]Congratulations! :tada:[/bold]")
            console.print(f"The word was [bold]{guess}[/bold].")
            break

        # Otherwise, filter and loop.
        try:
            words_arr = process_result(words_arr, guess, resp)
        except Exception as e:
            console.print(f"An internal error occurred: {e}", style="bold")
            logger.error("Error during process_result", exc_info=True)
            break


if __name__ == "__main__":
    app()
