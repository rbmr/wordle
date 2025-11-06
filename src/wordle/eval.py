import importlib
import importlib.resources
import logging
import time

import numpy as np

from wordle.pick import Strategy, pick_best_word, pick_letter_freq, pick_min_remaining
from wordle.solver import get_resp, process_result, to_str, words_to_arr, load_words

logger = logging.getLogger(__name__)

def _solve_game(
        all_candidates: np.ndarray,
        all_guesses: np.ndarray,
        first_guess_arr: np.ndarray,
        true_word_arr: np.ndarray,
        strategy: Strategy,
) -> int:
    """
    Simulates a single game for one true word and returns the guess count.

    Args:
        true_word_arr: (k,) array. The secret answer to find.
        all_candidates: (N, k) array. The full list of possible answers.
        all_guesses: (M, k) array. The full list of allowed guesses.
        strategy: The strategy function to use.
        first_guess_arr: (k,) array. The pre-computed first guess to use.

    Returns:
        The number of guesses it took to find the true word.
    """
    n_guesses = 1
    guess_arr = first_guess_arr
    current_candidates = all_candidates

    # Loop until the guess matches the true word
    while not np.array_equal(guess_arr, true_word_arr):
        # Get the response from the last guess
        resp = get_resp(true=true_word_arr, guess=guess_arr)

        # Filter the candidate list
        current_candidates = process_result(current_candidates, guess_arr, resp)
        n_guesses += 1

        # Determine the next guess
        guess_arr = pick_best_word(
            strategy,
            current_candidates,
            all_guesses
        )

    # The loop broke, meaning guess_arr == true_word_arr
    return n_guesses


def compute_expected_guesses(
        all_candidates: np.ndarray,
        all_guesses: np.ndarray,
        strategy: Strategy,
) -> float:
    """
    Computes the expected (average) number of guesses for a given
    strategy by simulating a game for every word in the candidate list.
    """

    # Get the first guess
    logger.info(f"Calculating first guess for {strategy.__name__}...")
    start_time = time.time()
    first_guess = pick_best_word(
        strategy,
        all_candidates,
        all_guesses
    )
    end_time = time.time()
    logger.info(
        f"First guess is {to_str(first_guess)}. "
        f"(Took {end_time - start_time:.2f}s)"
    )

    # Simulate the game for every candidate word
    total_guesses = 0
    n_candidates = all_candidates.shape[0]

    logger.info(f"Starting simulation for {n_candidates} words...")
    sim_start_time = time.time()

    for i in range(n_candidates):
        true_word = all_candidates[i, :]
        cost = _solve_game(
            all_candidates,
            all_guesses,
            first_guess,
            true_word,
            strategy,
        )

        total_guesses += cost

        if (i + 1) % 100 == 0:
            logger.info(f"  ...simulated {i + 1} / {n_candidates} words")

    sim_end_time = time.time()
    logger.info(
        f"Simulation complete. (Took {sim_end_time - sim_start_time:.2f}s)"
    )

    # 3. Return the average
    return total_guesses / n_candidates


def main():
    # Set up basic logging to see progress
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]",
    )

    # Load the default word list (similar to main.py)
    logger.info("Loading default word list...")
    words_list = []
    try:
        default_words_file = importlib.resources.files("wordle").joinpath(
            "words.txt"
        )
        with importlib.resources.as_file(default_words_file) as words_file_path:
            words_list = load_words(words_file_path, n_chars=5)
    except Exception as e:
        logger.error(f"Failed to load default word list: {e}", exc_info=True)
        return

    if not words_list:
        logger.error("No words loaded. Exiting.")
        return

    # Convert list to numpy array
    all_candidates_arr = words_to_arr(words_list)

    # Per your request, use the candidate list as the guess list for now
    all_guesses_arr = all_candidates_arr

    logger.info(f"Loaded {all_candidates_arr.shape[0]} words for evaluation.")

    # Test the "minimax" strategy
    avg_guesses_minimax = compute_expected_guesses(
        all_candidates_arr,
        all_guesses_arr,
        pick_min_remaining
    )
    print(f"Minimax Strategy E[Guesses]: {avg_guesses_minimax:.4f}")

    # Test the "letter freq" strategy
    avg_guesses_freq = compute_expected_guesses(
        all_candidates_arr,
        all_guesses_arr,
        pick_letter_freq
    )
    print(f"Letter Freq Strategy E[Guesses]: {avg_guesses_freq:.4f}")


if __name__ == "__main__":
    main()