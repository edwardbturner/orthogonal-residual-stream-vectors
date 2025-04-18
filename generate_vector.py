import torch
import transformer_lens  # type: ignore
from transformers import AutoTokenizer

from utils import melbo_ortho


def main(prompt: str):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and tokenizer
    model = transformer_lens.HookedTransformer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device=device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

    # Load existing vectors to make new vector orthogonal to them
    existing_vectors = torch.load("vector_files/this one gives python code-0-RUN-2.pt")

    # Create initial random vector
    d_model = model.cfg.d_model
    pretrained_theta = torch.randn(d_model, device=device)

    # Generate the steering vector
    poetry_vector = melbo_ortho(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        pretrained_theta=pretrained_theta,
        target_later=torch.zeros(d_model, device=device),  # You might want to adjust this
        make_ortho_to=existing_vectors,
        refine_epochs=1000,
        enforce_orthogonality=True,
    )

    # Save the vector
    torch.save(poetry_vector, "vector_files/this one writes poetry-0-RUN-1.pt")
    print("Saved poetry steering vector")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main(prompt="Write a beautiful poem about nature")
