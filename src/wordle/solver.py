import importlib
import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

B, G, Y = b"B", b"G", b"Y"


def load_default_words(filename: str, n_chars: int = 5) -> List[str]:
    """
    Loads a default word list from the package data using importlib.resources.
    """
    logger.info(f"Loading default word list: {filename}")
    try:
        default_file = importlib.resources.files("wordle").joinpath("words", filename)
        with importlib.resources.as_file(default_file) as file_path:
            return load_words(file_path, n_chars=n_chars)
    except FileNotFoundError:
        logger.error(f"Error: Default '{filename}' not found within the package.")
        return []
    except Exception as e:
        logger.error(f"An error occurred loading '{filename}': {e}", exc_info=True)
        return []


def load_words(words_file: Path, n_chars: int = 5) -> list[str]:
    """
    Loads words from a file, verifying each word has n_chars.

    Assumes words are separated by whitespace (spaces, tabs, newlines).
    """
    valid_words = set()
    try:
        with words_file.open("r") as f:
            all_text = f.read()
            all_words = all_text.split()
            for word in all_words:
                if len(word) == n_chars:
                    valid_words.add(word.upper())
                else:
                    logger.warning(
                        f"Unexpected number of characters: {word}, expected {n_chars}"
                    )
    except FileNotFoundError:
        logger.error(f"The file '{words_file}' was not found.")
        return []  # Return an empty list on error
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []  # Return an empty list on error
    logger.info(f"Successfully loaded {len(valid_words)} from {words_file}")
    return sorted(valid_words)


def words_to_arr(words: list[str]) -> np.ndarray:
    """Converts a list of equal length python strings to a numpy array."""
    assert len(words) > 0
    word_len = len(words[0])
    assert all(len(word) == word_len for word in words)
    char_array = np.array([list(word) for word in words], dtype="S1")
    return char_array


def to_str(word: np.ndarray) -> str:
    return word.tobytes().decode("utf-8")


def process_result(
    words: np.ndarray, guess: np.ndarray, resp: np.ndarray
) -> np.ndarray:
    """
    Processes responses to a guess by filtering the words list using NumPy.

    'words' is a (n_samples, n_chars) NumPy array of bytes/chars ('S1').
    'guess' is the guessed word (e.g., "arise").
    'resp' is the Wordle response (e.g., "BYYGB").
    """
    assert set(resp).issubset({B, G, Y})
    assert guess.ndim == resp.ndim == 1
    assert guess.shape[0] == resp.shape[0] == words.shape[1]

    mask = np.ones(len(words), dtype=bool)

    for char_byte in np.unique(guess):
        guess_indices = np.where(guess == char_byte)[0]
        char_resps = resp[guess_indices]

        # Handle Greens
        green_indices = guess_indices[char_resps == G]
        for idx in green_indices:
            mask &= words[:, idx] == char_byte

        # Handle Yellows
        yellow_indices = guess_indices[char_resps == Y]
        for idx in yellow_indices:
            mask &= words[:, idx] != char_byte

        # Handle Blacks
        black_indices = guess_indices[char_resps == B]
        for idx in black_indices:
            mask &= words[:, idx] != char_byte

        # Handle Counts
        n_green = len(green_indices)
        n_yellow = len(yellow_indices)
        n_present = n_green + n_yellow

        is_black = np.any(char_resps == B)
        char_counts_in_words = np.sum(words == char_byte, axis=1)
        if is_black:
            # A 'B' means the count of this char is *exactly* n_present.
            mask &= char_counts_in_words == n_present
        else:
            # No 'B's means the count of this char is *at least* n_present.
            mask &= char_counts_in_words >= n_present

    return words[mask]


def get_indices(arr: np.ndarray) -> np.ndarray:
    """
    Converts a NumPy array of bytes ('S1') to a NumPy array
    of uint indices (0-25).
    """
    # .view(np.uint8) is a zero-copy operation, very fast
    uint_arr = arr.view(np.uint8)
    assert np.all((uint_arr >= 65) & (uint_arr <= 90))

    # Convert ASCII (65-90) to indices (0-25)
    return uint_arr - 65


def get_counts_1dim(indices_arr: np.ndarray) -> np.ndarray:
    """
    Calculates letter counts (0-25) from a 1d array of indices.
    """
    assert indices_arr.ndim == 1
    assert np.all(
        (indices_arr >= 0) & (indices_arr <= 25)
    ), "Indices are not in the expected [0, 25] range"
    return np.bincount(indices_arr, minlength=26)


def get_counts_2dim(indices_arr: np.ndarray) -> np.ndarray:
    """
    Calculates letter counts (0-25) from a 2d array of indices.
    """
    assert indices_arr.ndim == 2
    assert np.all(
        (indices_arr >= 0) & (indices_arr <= 25)
    ), "Indices are not in the expected [0, 25] range"

    N, k = indices_arr.shape
    assert k < 256
    counts_arr = np.zeros((N, 26), dtype=np.uint8)
    row_indices = np.arange(N)[:, None]  # Shape (N, 1)

    # Broadcasts row_indices to (N, k) and increments
    # counts_arr[r, c] for each (r, c) pair.
    np.add.at(counts_arr, (row_indices, indices_arr), 1)
    return counts_arr


def get_all_resp(
    candidates: np.ndarray,
    candidates_idx: np.ndarray,
    true_counts_all: np.ndarray,
    guess: np.ndarray,
) -> np.ndarray:

    # Precompute relevant values
    N, k = candidates.shape
    true_counts = true_counts_all.copy()
    guess_idx = get_indices(guess)

    # Set blacks
    full_resp = np.full((N, k), B, dtype="S1")

    # Set greens
    green_mask = candidates == guess
    full_resp[green_mask] = G

    # Decrement counts for the green letters
    green_counts_to_sub = np.zeros_like(true_counts_all)
    green_rows, green_cols = np.where(green_mask)
    green_letter_indices = candidates_idx[green_rows, green_cols]
    np.add.at(green_counts_to_sub, (green_rows, green_letter_indices), 1)
    true_counts -= green_counts_to_sub

    # Yellow pass
    for i in range(k):
        letter_idx = guess_idx[i]

        # Find words that are NOT green at this position
        not_green_mask = ~green_mask[:, i]  # (N,)

        # Find words that HAVE this letter still available
        has_letter_mask = true_counts[:, letter_idx] > 0  # (N,)

        # Combine: Can be yellow if not green AND letter is available
        yellow_mask = not_green_mask & has_letter_mask  # (N,)

        # Set those positions to Yellow
        full_resp[yellow_mask, i] = Y

        # And decrement the count *only for those words*
        true_counts[yellow_mask, letter_idx] -= 1

    return full_resp


def get_resp(true: np.ndarray, guess: np.ndarray) -> np.ndarray:
    """
    Generates a Wordle response given a true word and a guess.
    """
    n_chars = true.shape[0]
    assert guess.shape[0] == n_chars

    # Promote 1D inputs to a 2D batch of size N=1
    candidates = true[None, :]  # (k,) -> (1, k)
    candidates_idx = get_indices(candidates)  # (1, k)
    true_counts_all = get_counts_2dim(candidates_idx)  # (1, 26)
    all_resps = get_all_resp(
        candidates, candidates_idx, true_counts_all, guess
    )  # (1, k)
    return all_resps[0, :]  # (1, k) -> (k,)
