# Wordle Solver CLI

A simple, interactive command-line tool to help you solve Wordle puzzles. It suggests the best word to guess next and filters its word list based on the Green, Yellow, and Black responses you provide.

This tool is built with:
* **Typer** for a clean command-line interface.
* **Rich** for beautiful, interactive terminal output.
* **NumPy** for high-performance word filtering.


---

## 🚀 Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast dependency and environment management.

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/wordle.git](https://github.com/your-username/wordle.git)
    cd wordle
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    uv venv
    source .venv/bin/activate
    # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install the project and its dependencies:**
    ```sh
    uv pip install .
    ```

---

## 🎮 How to Use

Once installed, simply run the `wordle-solver` command in your terminal.

```sh
wordle-solver
````

The solver will start, load the default word list (`words.txt`), and suggest an initial guess (usually "RAISE" or a similar high-frequency word).

1.  Enter the suggested word (or your own guess) into the actual Wordle game.
2.  Enter the color response you received (e.g., `GYBBB`) into the solver.
      * **G** = Green
      * **Y** = Yellow
      * **B** = Black
3.  The solver will filter its word list and suggest the next best word.
4.  Repeat until you solve the puzzle\!

### Using a Custom Word List

You can provide your own text file of valid 5-letter words by passing it as an argument:

```sh
wordle-solver /path/to/my-custom-words.txt
```

-----

## 🧠 How It Works

The solver's strategy is based on two main functions:

  * **`process_result` (Filtering):** This function takes your guess and the `GYB` response. It uses NumPy to create boolean masks that efficiently filter the entire word list, eliminating any words that do not match the clues you provided. It correctly handles complex cases like duplicate letters, multiple yellows, and "black" clues that fix the exact count of a letter.

  * **`pick_best_word` (Suggesting):** To suggest the best next guess, the solver calculates the frequency of all letters *remaining* in the possible word list. It then scores each remaining word based on the sum of frequencies of its **unique** letters. The word with the highest score is suggested, as it's the most likely to provide new information and narrow down the possibilities.

-----

## 🛠️ Development

To set up an environment for development (including tools like `black`, `isort`, and `pytest` specified in `pyproject.toml`):

1.  Follow the installation steps 1 and 2 to create and activate the `uv` virtual environment.

2.  Install the project in **editable mode** with the `dev` dependencies:

    ```sh
    uv pip install -e ".[dev]"
    ```

You can then run the formatters and other dev tools:

```sh
# Format code
black .
isort .

# Run tests
pytest
```