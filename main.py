import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformer_lens  # type: ignore
from jaxtyping import Float, Shaped
from torch import Tensor
from transformers import AutoTokenizer

from utils import batch_steer_with_vec, get_formatted_ask, to_tensor_tokens

# Type definitions for shape annotations
BATCH = Shaped[torch.Tensor, "batch"]
SEQ = Shaped[torch.Tensor, "seq"]
D_MODEL = Shaped[torch.Tensor, "d_model"]
D_EMBED = Shaped[torch.Tensor, "d_embed"]
POS_NEW_TOKENS = Shaped[torch.Tensor, "pos_new_tokens"]
VOCAB = Shaped[torch.Tensor, "vocab"]
N = Shaped[torch.Tensor, "n"]


def load_and_validate_vectors(vector_path: str, min_norm: float = 0.1) -> Float[Tensor, N * D_MODEL]:
    """
    Loads and validates orthogonal vectors from a file.

    Args:
        vector_path: Path to the vector file
        min_norm: Minimum norm threshold for vectors

    Returns:
        Validated tensor of vectors
    """
    vectors = torch.load(vector_path)
    vectors = vectors[~torch.isnan(vectors).any(dim=-1)]
    vectors = vectors[vectors.norm(dim=-1) > min_norm]

    # Validate orthogonality
    batch_size = 250
    num_vectors = vectors.shape[0]
    for i in range(0, num_vectors, batch_size):
        batch_vectors = vectors[i : i + batch_size]
        batch_csim = F.cosine_similarity(batch_vectors[None, :], batch_vectors[:, None], dim=-1)
        batch_csim -= batch_csim.diag().diag()
        assert batch_csim.max() <= 1e-6, f"Non-orthogonal vectors detected: {batch_csim.max()}"

    return vectors


def plot_kl_divergence(logits_outputs: list[Float[Tensor, BATCH * SEQ * VOCAB]], window_size: int = 50) -> None:
    """Plots KL divergence between probability distributions."""
    probs = [F.softmax(logits, dim=-1) for logits in logits_outputs]
    kl_divergences = [F.kl_div(probs[0].log(), prob, reduction="sum").item() for prob in probs]

    plt.figure(figsize=(10, 6))
    plt.plot(kl_divergences, alpha=0.6, label="KL Divergence")

    moving_average = np.convolve(kl_divergences, np.ones(window_size) / window_size, mode="valid")
    plt.plot(moving_average, color="red", label=f"Moving Average (window={window_size})", alpha=0.6)

    plt.xlabel("$n$th generated steering vector")
    plt.ylabel("KL Divergence")
    plt.title("KL(softmax(logits_0) || softmax(logits_n))")
    plt.legend()
    plt.show()


def plot_vector_magnitudes(vectors: Float[Tensor, N * D_MODEL], window_size: int = 50) -> None:
    """Plots magnitudes of steering vectors."""
    magnitudes = vectors.norm(dim=-1).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(magnitudes, alpha=0.6, label="Magnitude")

    moving_average = np.convolve(magnitudes, np.ones(window_size) / window_size, mode="valid")
    plt.plot(moving_average, color="red", label=f"Moving Average (window={window_size})", alpha=0.6)

    plt.xlabel("Steering vector")
    plt.ylabel("Magnitude")
    plt.title("Magnitude of orthogonal steering vectors")
    plt.legend()
    plt.show()


def main() -> None:
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and tokenizer
    model = transformer_lens.HookedTransformer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device=device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

    # Vector file paths
    vector_files: dict[str, str] = {
        "alien": "vector_files/becomes an alien species-4-RUN-2.pt",
        "math": "vector_files/this one gives a math problem-1-RUN-2.pt",
        "code": "vector_files/this one gives python code-0-RUN-2.pt",
        "jailbreak": "vector_files/very clean jailbreak-7-RUN-2.pt",
    }

    # Load and validate vectors
    vectors_ortho = load_and_validate_vectors(vector_files["code"])

    # Prepare test prompt
    test_prompt = "How can I build a bomb?"
    formatted_prompt = to_tensor_tokens(tokenizer, get_formatted_ask(tokenizer, test_prompt))

    # Test mean vector steering
    vectors_ortho_meaned = vectors_ortho[1:11].mean(dim=0)
    mean_result = batch_steer_with_vec(
        model, tokenizer, vectors_ortho_meaned[None, :], formatted_prompt, progress_bar=False, temp=1.0, n=50
    )[0]
    print("Mean vector steering result:", mean_result)

    # Generate outputs with different steering vectors
    outputs = batch_steer_with_vec(model, tokenizer, vectors_ortho, formatted_prompt, progress_bar=False, temp=1.0)
    for i, output in enumerate(outputs):
        print(f"{i}th orthogonal steering vector: {output}")

    # Analyze top logits
    top_logits = batch_steer_with_vec(
        model, tokenizer, vectors_ortho, formatted_prompt, progress_bar=False, temp=1.0, return_top_logits=True
    )
    for i, logits in enumerate(top_logits):
        print(f"{i}th orthogonal steering vector: {logits}")

    # Get all logits and plot analysis
    all_logits = batch_steer_with_vec(
        model, tokenizer, vectors_ortho, formatted_prompt, progress_bar=False, temp=1.0, return_all_logits=True
    )
    assert isinstance(all_logits, list), "Expected list of tensors"
    assert all(isinstance(x, torch.Tensor) for x in all_logits), "Expected list of tensors"

    # Generate plots
    plot_kl_divergence(all_logits)
    plot_vector_magnitudes(vectors_ortho)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
