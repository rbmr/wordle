import importlib.resources
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from wordle.pick import pick_best_word, pick_min_remaining
from wordle.solver import (
    load_default_words,
    load_words,
    process_result,
    to_str,
    words_to_arr,
)

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
    """Get a valid guess from the user"""
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
    """Get a valid wordle response from the user"""
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


@app.command()
def main(
    candidates_file: Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a custom candidate (answer) list file.",
    ),
    guesses_file: Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a custom *full* list of valid guesses.",
    ),
):
    """
    Runs the command-line Wordle solver.
    """
    # Load candidates
    if candidates_file is not None:
        if not candidates_file.is_file():
            raise ValueError("Candidate file does not exist.")
        console.print(f"Using custom candidate list: [dim]{candidates_file}[/dim]")
        candidates_list = load_words(candidates_file, n_chars=5)
    else:
        console.print(f"Using default candidate list.")
        candidates_list = load_default_words("candidates.txt", n_chars=5)

    # Load valid words
    if guesses_file is not None:
        if not guesses_file.is_file():
            raise ValueError("Valid words file does not exist.")
        console.print(f"Using custom valid guesses list: [dim]{guesses_file}[/dim]")
        guesses_list = load_words(guesses_file, n_chars=5)
    else:
        console.print(f"Using default valid guesses list.")
        guesses_list = load_default_words("guesses.txt", n_chars=5)

    # Ensure all candidates may be guessed
    candidates_set = set(candidates_list)
    guesses_set = set(guesses_list)
    missing_words = candidates_set.difference(guesses_set)
    if missing_words:
        console.print(
            f"Adding {len(missing_words)} missing candidates to list of valid guesses."
        )
        guesses_list.extend(missing_words)

    # Ensure candidates exist.
    if not candidates_list:
        console.print("No candidates exist. Exiting.", style="bold")
        raise typer.Exit(code=1)

    # Convert to NumPy
    candidates_arr = words_to_arr(candidates_list)
    guesses_arr = words_to_arr(guesses_list)
    n_chars = candidates_arr.shape[1]
    GREEN_WIN = "G" * n_chars

    console.print(
        f"Starting Wordle Solver CLI (Loaded {len(candidates_list)} candidates, "
        f"{len(guesses_list)} valid guesses)",
        style="bold",
    )
    while True:

        # Determine number of candidates remaining
        n_remaining = len(candidates_arr)
        if n_remaining == 0:
            console.print("\n[bold]No possible words match the given clues.[/bold]")
            break
        console.print(f"\n[bold]{n_remaining}[/bold] possible words remaining")

        # Display the remaining words
        max_words = 128
        display_words = min(n_remaining, max_words)
        decoded_words = [to_str(candidates_arr[i, :]) for i in range(display_words)]

        words_text = Text(", ").join(Text(w, style="dim") for w in decoded_words)
        if n_remaining > max_words:
            words_text.append(
                f" and {n_remaining - max_words} more...", style="italic dim"
            )
        console.print(words_text)

        # Suggest a word to guess
        suggested_arr = pick_best_word(
            strategy=pick_min_remaining,
            candidates=candidates_arr,
            guesses=guesses_arr,
        )
        suggested_word = to_str(suggested_arr)
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
            guess_arr = np.array(list(guess), dtype="S1")
            resp_arr = np.array(list(resp), dtype="S1")
            candidates_arr = process_result(candidates_arr, guess_arr, resp_arr)
        except Exception as e:
            console.print(f"An internal error occurred: {e}", style="bold")
            logger.error("Error during process_result", exc_info=True)
            break


if __name__ == "__main__":
    app()
