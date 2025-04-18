## Credits

This project is a quick clean of the original implementation here:
[melbo-ortho](https://github.com/g-w1/melbo-ortho)
from the LessWrong post here:
[I Found >800 Orthogonal Write Code Steering](https://www.lesswrong.com/posts/CbSEZSpjdpnvBcEvc/i-found-greater-than-800-orthogonal-write-code-steering)


# Orthogonal Residual Stream Vectors

This project implements and analyzes orthogonal residual stream vectors for language model steering. It provides tools for:
- Generating and validating orthogonal steering vectors
- Applying steering vectors to language models
- Analyzing the effects of steering on model behavior
- Visualizing vector properties and model behavior

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project is organized into two main files:
- `utils.py`: Contains utility functions for vector manipulation and model interaction
- `main.py`: Contains the main execution flow and analysis code

To run the analysis:
```bash
python main.py
```

This will:
1. Load and validate orthogonal vectors
2. Apply steering vectors to a test prompt
3. Generate and analyze model outputs
4. Create visualizations of vector properties and model behavior

## Vector Files

The project expects vector files in the following format:
- `becomes an alien species-4-RUN-2.pt`
- `this one gives a math problem-1-RUN-2.pt`
- `this one gives python code-0-RUN-2.pt`
- `very clean jailbreak-7-RUN-2.pt`

These files should contain torch tensors of steering vectors.