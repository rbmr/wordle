from typing import Optional, Callable, Counter

import numpy as np
import logging

from wordle.solver import get_resp, get_indices, get_all_resp, get_counts_2dim, get_counts_1dim

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
        logger.info("Only one word remaining, picking index 0.")
        return candidates[0, :]
    if n_candidates == 2:
        logger.info("Only two words remaining, picking index 0.")
        return candidates[0, :]
    # If no separate guess list is provided, use the candidates.
    if guesses is None:
        guesses = candidates
    # Use the strategy to find the best guess
    return strategy(candidates, guesses)


def pick_letter_freq(
        candidates: np.ndarray,
        guesses: np.ndarray
) -> np.ndarray:
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

def pick_min_remaining(
        candidates: np.ndarray,
        guesses: np.ndarray
) -> np.ndarray:
    """
    Picks the word from `guess_list` that minimizes the expected size
    of the `candidates` list for the next turn.
    """
    n_candidates, k = candidates.shape
    n_guesses, k2 = guesses.shape
    assert k == k2
    assert n_candidates > 0
    assert n_guesses > 0

    min_score = float('inf')
    best_guess = None  # Will be overwritten

    # Cache relevant matrices
    candidates_idx = get_indices(candidates)
    true_counts_all = get_counts_2dim(candidates_idx)

    # Outer loop: Iterate through each *potential guess*
    for i in range(n_guesses):

        guess_arr = guesses[i, :]

        # Call the new vectorized function
        # This one call replaces the entire inner Python loop
        all_resps = get_all_resp(
            candidates,
            candidates_idx,
            true_counts_all,
            guess_arr
        )

        # Now we have an (N, k) array of responses. We need to
        # find the counts of each unique response row.

        # This is a NumPy trick to view rows as single, hashable items
        # It's the C-level equivalent of `Counter(resp.tobytes() for resp in all_resps)`
        # It's extremely fast.
        void_view = np.ascontiguousarray(all_resps).view(
            np.dtype((np.void, k * all_resps.dtype.itemsize))
        )

        # `np.unique` finds all unique rows and their counts
        _, counts = np.unique(void_view.ravel(), return_counts=True)

        # Calculate the score from the partition sizes (counts)
        current_score = np.sum(counts ** 2)

        # Lower score is better
        if current_score < min_score:
            min_score = current_score
            best_guess = guess_arr

    return best_guess