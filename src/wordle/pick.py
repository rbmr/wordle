import logging
from typing import Callable, Optional

import numpy as np

from wordle.solver import get_all_resp, get_counts_1dim, get_counts_2dim, get_indices

logger = logging.getLogger(__name__)

Strategy = Callable[[np.ndarray, np.ndarray], np.ndarray]


def pick_best_word(
    strategy: Strategy,
    candidates: np.ndarray,
    guesses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Selects the best word to guess using a provided strategy function.
    """
    n_candidates = candidates.shape[0]
    if n_candidates == 0:
        raise ValueError("Candidate array is empty")
    if n_candidates == 1:
        logger.debug("Only one word remaining, picking index 0.")
        return candidates[0, :]
    if n_candidates == 2:
        logger.debug("Only two words remaining, picking index 0.")
        return candidates[0, :]
    # If no separate guess list is provided, use the candidates.
    if guesses is None:
        guesses = candidates
    # Use the strategy to find the best guess
    return strategy(candidates, guesses)


def pick_letter_freq(candidates: np.ndarray, guesses: np.ndarray) -> np.ndarray:
    """
    Picks the word from `guesses` that contains the most
    frequent *unique* letters from the `candidates` list.
    """
    # Convert ASCII (65-90) to indices (0-25)
    candidates_idx = get_indices(candidates)
    guesses_idx = get_indices(guesses)

    # Calculate frequency of all letters
    letter_frequencies = get_counts_1dim(candidates_idx.flatten())

    # Sort letters within each guess word
    sorted_guesses_idx = np.sort(guesses_idx, axis=1)

    # Create a mask to find unique letters
    unique_mask = np.full(sorted_guesses_idx.shape, True, dtype=bool)
    unique_mask[:, 1:] = sorted_guesses_idx[:, 1:] != sorted_guesses_idx[:, :-1]

    # Get the frequency score for each letter in each word
    sorted_guess_scores = letter_frequencies[sorted_guesses_idx]

    # Zero out scores for duplicate letters, then sum
    final_scores = np.sum(sorted_guess_scores * unique_mask, axis=1)

    # Find the index of the best guess
    best_idx = np.argmax(final_scores)

    return guesses[best_idx, :]


def pick_min_remaining(candidates: np.ndarray, guesses: np.ndarray) -> np.ndarray:
    """
    Picks the word from `guess_list` that minimizes the expected size
    of the `candidates` list for the next turn.
    """
    n_candidates, k = candidates.shape
    n_guesses, k2 = guesses.shape
    assert k == k2
    assert n_candidates > 0
    assert n_guesses > 0

    min_score = float("inf")
    best_guess = None  # Will be overwritten

    # Cache relevant matrices
    candidates_idx = get_indices(candidates)
    true_counts_all = get_counts_2dim(candidates_idx)

    # Outer loop: Iterate through each *potential guess*
    for i in range(n_guesses):

        guess_arr = guesses[i, :]

        # Get all responses for the guess, across all candidates
        all_resps = get_all_resp(candidates, candidates_idx, true_counts_all, guess_arr)

        # NumPy trick to view rows as single, hashable items
        void_view = np.ascontiguousarray(all_resps).view(
            np.dtype((np.void, k * all_resps.dtype.itemsize))
        )

        # Find the counts for each unique response row.
        _, counts = np.unique(void_view.ravel(), return_counts=True)

        # Calculate the score from the partition sizes (counts)
        current_score = np.sum(counts**2)

        # Lower score is better
        if current_score < min_score:
            min_score = current_score
            best_guess = guess_arr

    return best_guess
