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