import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from pydantic import BaseModel

from wordle.pick import Strategy, pick_best_word, pick_letter_freq, pick_min_remaining
from wordle.solver import (
    get_resp,
    load_default_words,
    load_words,
    process_result,
    to_str,
    words_to_arr,
)

logger = logging.getLogger(__name__)


class RunStats(BaseModel):
    """Holds the timing and statistical results of an evaluation run."""

    first_guess_time_s: float
    total_sim_time_s: float
    avg_game_time_s: float
    mean_guesses: float
    variance_guesses: float


class EvalResult(BaseModel):
    """Top-level model for all evaluation results."""

    strategy_name: str
    n_candidates: int
    n_total_guesses: int
    first_guess: str
    stats: RunStats
    distribution_counts: List[int]
    guesses_per_word: Dict[str, int]


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
        assert len(current_candidates) > 0
        n_guesses += 1

        # Determine the next guess
        guess_arr = pick_best_word(strategy, current_candidates, all_guesses)

    # The loop broke, meaning guess_arr == true_word_arr
    return n_guesses


def evaluate_strategy(
    all_candidates: np.ndarray,
    all_guesses: np.ndarray,
    strategy: Strategy,
) -> EvalResult:
    """
    Evaluates a strategy by simulating a game for every word in the list of candidates.
    """
    # Setup
    logger.info(f"Evaluating {strategy.__name__}")
    n_candidates = all_candidates.shape[0]
    n_total_guesses = all_guesses.shape[0]

    # Get the first guess
    logger.info("Calculating first guess...")
    start_time_first_guess = time.perf_counter()
    first_guess_arr = pick_best_word(strategy, all_candidates, all_guesses)
    end_time_first_guess = time.perf_counter()
    first_guess_time_s = end_time_first_guess - start_time_first_guess
    first_guess_str = to_str(first_guess_arr)
    logger.info(f"First guess is {first_guess_str}. (Took {first_guess_time_s:.2f}s)")

    # Simulate all games using first guess
    guesses_per_word: Dict[str, int] = {}
    game_times_s = []

    logger.info(f"Starting simulation for {n_candidates} words...")
    total_sim_start_time = time.perf_counter()

    for i in range(n_candidates):
        true_word_arr = all_candidates[i, :]
        true_word_str = to_str(true_word_arr)

        game_start_time = time.perf_counter()
        cost = _solve_game(
            all_candidates,
            all_guesses,
            first_guess_arr,
            true_word_arr,
            strategy,
        )
        game_end_time = time.perf_counter()

        guesses_per_word[true_word_str] = cost
        game_times_s.append(game_end_time - game_start_time)

        if (i + 1) % 100 == 0:
            logger.info(f"  ...simulated {i + 1} / {n_candidates} words")

    total_sim_end_time = time.perf_counter()
    total_sim_time_s = total_sim_end_time - total_sim_start_time
    logger.info(f"Simulation complete. (Took {total_sim_time_s:.2f}s)")

    # Calculate statistics
    counts_arr = np.array(list(guesses_per_word.values()))
    times_arr = np.array(game_times_s)

    stats_model = RunStats(
        first_guess_time_s=first_guess_time_s,
        total_sim_time_s=total_sim_time_s,
        avg_game_time_s=float(np.mean(times_arr)),
        mean_guesses=float(np.mean(counts_arr)),
        variance_guesses=float(np.var(counts_arr)),
    )

    results_model = EvalResult(
        strategy_name=strategy.__name__,
        n_candidates=n_candidates,
        n_total_guesses=n_total_guesses,
        first_guess=first_guess_str,
        stats=stats_model,
        distribution_counts=np.bincount(counts_arr).tolist(),
        guesses_per_word=guesses_per_word,
    )

    return results_model


def get_words_hash(candidates_list: List[str], guesses_list: List[str]) -> str:
    """Creates a unique hash for the combination of two word lists."""
    candidates_str = ",".join(sorted(candidates_list))
    guesses_str = ",".join(sorted(guesses_list))

    hasher = hashlib.sha256()
    hasher.update(candidates_str.encode("utf-8"))
    hasher.update(guesses_str.encode("utf-8"))

    return hasher.hexdigest()[:12]


def get_dir_name(
    strategy_name: str, n_candidates: int, n_guesses: int, words_hash: str
) -> str:
    return f"{strategy_name}_{n_candidates}_{n_guesses}_{words_hash}"


def save_results_json(results_data: EvalResult, output_file: Path):
    """Saves the Pydantic model to results.json."""
    assert output_file.suffix == ".json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(results_data.model_dump_json(indent=2))


def create_shareable_image(results_data: EvalResult, output_file: Path):
    """
    Creates a single sharable image file from the EvalResult model.
    """
    # Data Prep (using model attributes)
    strategy_name = results_data.strategy_name
    stats = results_data.stats

    # [0, count_1, count_2, ...]
    dist_counts_list = results_data.distribution_counts
    n_candidates = results_data.n_candidates

    # Prep for bar plot
    max_guess = len(dist_counts_list) - 1
    x_values = np.arange(1, max_guess + 1)

    # Get counts by slicing the list (skip index 0)
    y_counts = np.array(dist_counts_list[1:])
    y_freq = y_counts / n_candidates

    # Prep for table
    table_data = []
    # Loop from 1 up to max_guess
    for n_guess in x_values:
        # Get count from y_counts (which is already 1-indexed)
        table_data.append([f"Guesses = {n_guess}", f"{y_counts[n_guess - 1]:,}"])

    table_data.extend(
        [
            ["---", "---"],
            ["Total Candidates", f"{n_candidates:,}"],
            ["Total Guesses", f"{results_data.n_total_guesses:,}"],
            ["Mean Guesses", f"{stats.mean_guesses:.4f}"],
            ["Guess Variance", f"{stats.variance_guesses:.4f}"],
            ["---", "---"],
            ["First Guess Time", f"{stats.first_guess_time_s:.2f} s"],
            ["Avg. GameTime", f"{stats.avg_game_time_s:.4f} s"],  # Renamed for space
            ["Total Sim Time", f"{stats.total_sim_time_s:.2f} s"],
        ]
    )

    # --- Plotting (Unchanged from here) ---
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # 1. Bar Plot
    ax_plot = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(x_values)))
    ax_plot.bar(x_values, y_freq, color=colors, edgecolor="black", zorder=2)
    ax_plot.set_title(f"Guess Distribution: {strategy_name}", fontsize=16, pad=10)
    ax_plot.set_xlabel("Number of Guesses", fontsize=12)
    ax_plot.set_ylabel("Frequency", fontsize=12)
    ax_plot.set_xticks(x_values)
    ax_plot.set_ylim(0, 1.0)
    ax_plot.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax_plot.set_yticklabels([f"{x:.0%}" for x in ax_plot.get_yticks()])

    # 2. Table
    ax_table = fig.add_subplot(gs[0, 1])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("none")
        if table_data[row][0] == "---":
            cell.set_text_props(weight="bold")
            cell.set_height(0.1)
            cell.set_text(" " * 30)
            cell.set_edgecolor("black")
        elif col == 0:
            cell.set_text_props(weight="bold", ha="right", pad=5)
        else:
            cell.set_text_props(ha="left", pad=5)
    ax_table.set_title("Run Statistics", fontsize=16, pad=20)

    # Save Figure
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved shareable image to: {output_file}")


def main():
    # Set up basic logging to see progress
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]",
    )

    # Load the default word list (similar to main.py)
    logger.info("Loading default word list...")
    try:
        candidates_list = load_default_words("candidates.txt", n_chars=5)
        guesses_list = load_default_words("guesses.txt", n_chars=5)
    except Exception as e:
        logger.error(f"Failed to load default word lists: {e}", exc_info=True)
        return

    candidates_set = set(candidates_list)
    guesses_set = set(guesses_list)
    missing_words = candidates_set.difference(guesses_set)
    if missing_words:
        logger.info(
            f"Adding {len(missing_words)} missing candidates to list of valid guesses."
        )
        guesses_list.extend(missing_words)

    if not candidates_list:
        logger.error("No candidates loaded. Exiting.")
        return

    # Convert to numpy array
    all_candidates_arr = words_to_arr(candidates_list)
    all_guesses_arr = words_to_arr(guesses_list)
    n_candidates = all_candidates_arr.shape[0]
    n_guesses = all_guesses_arr.shape[0]
    logger.info(f"Loaded {n_candidates} candidates and {n_guesses} valid guesses.")

    # Get unique hash
    words_hash = get_words_hash(candidates_list, guesses_list)

    # Define strategies
    strategies_to_run = [
        pick_min_remaining,
        pick_letter_freq,
    ]

    # Evaluate each strategy
    for strategy in strategies_to_run:

        # Generate results
        results = evaluate_strategy(all_candidates_arr, all_guesses_arr, strategy)

        # Save results
        strategy_name = strategy.__name__
        output_dir_name = get_dir_name(
            strategy_name, n_candidates, n_guesses, words_hash
        )
        output_dir = Path(__file__).parent / "eval" / output_dir_name
        save_results_json(results, output_dir / "results.json")
        create_shareable_image(results, output_dir / "share.png")

    logger.info("All evaluations complete.")


if __name__ == "__main__":
    main()
