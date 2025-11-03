import logging
from collections import Counter

import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

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
    assert len(words) > 0
    word_len = len(words[0])
    assert word_len > 0
    assert all(len(word) == word_len for word in words)
    char_array = np.array([list(word) for word in words], dtype="S1")
    return char_array


def process_result(words: np.ndarray, guess: str, resp: str) -> np.ndarray:
    """
    Processes responses to a guess by filtering the words list using NumPy.

    'words' is a (n_samples, n_chars) NumPy array of bytes/chars ('S1').
    'guess' is the guessed word (e.g., "arise").
    'resp' is the Wordle response (e.g., "BYYGB").
    """
    assert set(resp).issubset({"B", "G", "Y"})
    assert len(resp) == len(guess) == words.shape[1]

    mask = np.ones(len(words), dtype=bool)
    guess_chars = np.array(list(guess), dtype="S1")
    resp_chars = np.array(list(resp), dtype="S1")
    B, G, Y = b"B", b"G", b"Y"

    for char_byte in np.unique(guess_chars):

        guess_indices = np.where(guess_chars == char_byte)[0]
        char_resps = resp_chars[guess_indices]

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


def get_word(words: np.ndarray, i: int) -> str:
    word_arr = words[i, :]
    return word_arr.tobytes().decode("utf-8")


def pick_best_word(words: np.ndarray) -> int:
    """
    Picks the word from the remaining list that contains the most
    frequent *unique* letters from that same list.
    """
    n_samples, n_chars = words.shape
    if n_samples == 0:
        logger.warning("No words remaining to pick from!")
        return -1  # Return a sentinel value
    if n_samples == 1:
        logger.info("Only one word remaining, picking index 0.")
        return 0  # Only one choice

    words_uint8 = words.view(np.uint8)
    assert np.all(
        (words_uint8 >= 65) & (words_uint8 <= 90)
    ), "Words contain non-uppercase A-Z characters!"

    # Convert ASCII (65-90) to indices (0-25)
    words_idx = words_uint8 - 65

    # Calculate frequency of all letters
    letter_frequencies = np.bincount(words_idx.flatten(), minlength=26)

    # Sort letters within each word (row-wise)
    sorted_words_idx = np.sort(words_idx, axis=1)
    unique_mask = np.full(sorted_words_idx.shape, True, dtype=bool)
    unique_mask[:, 1:] = sorted_words_idx[:, 1:] != sorted_words_idx[:, :-1]

    # Get the frequency score for each letter
    sorted_word_scores = letter_frequencies[sorted_words_idx]

    # Zero out scores for duplicate letters, then sum
    final_scores = np.sum(sorted_word_scores * unique_mask, axis=1)

    # Return the best index
    return np.argmax(final_scores)


def get_resp(true: str, guess: str) -> str:
    """
    Generates a Wordle response given a true word and a guess.
    """
    true = true.upper()
    guess = guess.upper()
    n_chars = len(true)

    assert len(guess) == n_chars, "Guess and true word must be the same length."

    # Initialize response as all 'B' (Black)
    resp = ["B"] * n_chars

    # Use Counter to track available letters in the true word.
    true_counts = Counter(true)

    # Greens take priority and "use up" a letter
    for i in range(n_chars):
        if guess[i] == true[i]:
            resp[i] = "G"
            true_counts[guess[i]] -= 1

    # Yellows can only use remaining, non-Green letters
    for i in range(n_chars):
        # Skip if it's already Green
        if resp[i] == "G":
            continue

        # Check if the guess letter is in the true word
        # and if we still have any of that letter available
        if guess[i] in true_counts and true_counts[guess[i]] > 0:
            resp[i] = "Y"
            true_counts[guess[i]] -= 1

    return "".join(resp)