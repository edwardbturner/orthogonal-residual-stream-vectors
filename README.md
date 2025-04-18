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

The project is organized into three main files:
- `utils.py`: Contains utility functions for vector manipulation and model interaction
- `main.py`: Contains the main execution flow and analysis code
- `generate_vector.py`: Script for generating new steering vectors

### Running Analysis
To run the analysis:
```bash
python main.py
```

This will:
1. Load and validate orthogonal vectors
2. Apply steering vectors to a test prompt
3. Generate and analyze model outputs
4. Create visualizations of vector properties and model behavior

### Generating New Vectors
To create a new steering vector:
```bash
python generate_vector.py
```

You can modify `generate_vector.py` to:
- Change the target behavior by modifying the prompt
- Adjust the training parameters (epochs, learning rate, etc.)
- Make the vector orthogonal to different existing vectors

## Vector Files
These files should contain torch tensors of steering vectors. You can create new vectors for different behaviors by:
1. Modifying the prompt in `generate_vector.py`
2. Running the generation script
3. Adding the new vector to the `vector_files` dictionary in `main.py`

## License

MIT License - See [LICENSE](LICENSE) for details.