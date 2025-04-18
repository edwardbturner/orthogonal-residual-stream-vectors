from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int, Shaped
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from transformer_lens import HookedTransformer  # type: ignore
from transformer_lens.hook_points import HookPoint  # type: ignore
from transformers import PreTrainedTokenizer

# Type definitions for shape annotations
BATCH = Shaped[torch.Tensor, "batch"]
SEQ = Shaped[torch.Tensor, "seq"]
D_MODEL = Shaped[torch.Tensor, "d_model"]
D_EMBED = Shaped[torch.Tensor, "d_embed"]
POS_PLUS_NEW_TOKENS = Shaped[torch.Tensor, "pos_plus_new_tokens"]
VOCAB = Shaped[torch.Tensor, "vocab"]
N = Shaped[torch.Tensor, "n"]


def auto_forward_n_times(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizer,
    base_text: Union[str, Float[torch.Tensor, BATCH * SEQ * D_EMBED]],
    change: Tuple[int, Float[torch.Tensor, D_MODEL], Optional[List[int]]],
    n: int,
    verbose: bool = False,
) -> List[str]:
    """
    Performs forward passes with residual stream modifications.

    Args:
        model: The transformer model
        tokenizer: The tokenizer instance
        base_text: Input text or tensor
        change: Tuple of (layer, vector, sequence_positions)
        n: Number of forward passes
        verbose: Whether to show progress

    Returns:
        List of generated texts
    """
    layer, vec, add_to_seq = change
    assert vec is not None

    def resid_stream_addition_hook(
        value: Float[torch.Tensor, BATCH * SEQ * D_MODEL], hook: HookPoint
    ) -> Float[torch.Tensor, BATCH * SEQ * D_MODEL]:
        assert vec is not None
        if add_to_seq is not None:
            for seq in add_to_seq:
                value[:, seq, :] = value[:, seq, :] + vec[None, :]
            return value
        else:
            return value + vec[None, None, :]

    model.add_hook(name=f"blocks.{layer}.hook_resid_post", hook=resid_stream_addition_hook)

    if isinstance(base_text, str):
        changed_text = model.generate(base_text, max_new_tokens=n, do_sample=False, verbose=verbose)
    else:
        output: Int[torch.Tensor, BATCH * POS_PLUS_NEW_TOKENS] = model.generate(
            base_text, max_new_tokens=n, do_sample=False, verbose=verbose
        )
        changed_text = [tokenizer.decode(output[b]) for b in range(output.shape[0])]

    model.reset_hooks()
    return changed_text


def get_formatted_ask(tokenizer: PreTrainedTokenizer, text: str, add_generation_prompt: bool = True) -> str:
    """
    Formats text for chat completion using the tokenizer's chat template.
    """
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def batch_steer_with_vec(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizer,
    vecs: Float[torch.Tensor, BATCH * D_MODEL],
    single_prompt: torch.Tensor,
    return_layer_16: bool = False,
    return_top_logits: bool = False,
    return_all_logits: bool = False,
    progress_bar: bool = True,
    temp: float = 0,
    n: int = 50,
) -> Union[torch.Tensor, List[str], List[Any]]:
    """
    Applies steering vectors in batches to generate text or analyze model behavior.
    """
    assert sum([return_layer_16, return_top_logits, return_all_logits]) <= 1
    dev = next(model.parameters()).device
    vecs_dataset = TensorDataset(vecs)
    results: List[Any] = []
    f: Callable[[Any], Any] = tqdm if progress_bar else lambda x: x

    for vecs_batch in f(DataLoader(vecs_dataset, batch_size=256)):
        vecs_batch = vecs_batch[0].to(dev)
        prompt_batch = torch.tensor(single_prompt).unsqueeze(0).repeat(vecs_batch.shape[0], 1)
        model.reset_hooks()

        def resid_stream_addition_hook(
            value: Float[torch.Tensor, BATCH * SEQ * D_MODEL], hook: HookPoint
        ) -> Float[torch.Tensor, BATCH * SEQ * D_MODEL]:
            return value + vecs_batch[:, None, :]

        model.add_hook("blocks.8.hook_resid_post", resid_stream_addition_hook)

        if return_layer_16:
            l16_out = model(prompt_batch, stop_at_layer=17)
            results.append(l16_out)
        elif return_top_logits:
            with torch.no_grad():
                logits = model.forward(prompt_batch)
                k = 10
                top_k = torch.topk(logits[:, -1, :], k, dim=-1)
                top_10_logits = top_k.values.cpu()
                top_10_indices = top_k.indices.cpu()
                for i in range(vecs_batch.shape[0]):
                    results.append(
                        f"\n\tlogits: {top_10_logits[i]}\n\tindices: {tokenizer.batch_decode(top_10_indices[i])}"
                    )
        elif return_all_logits:
            with torch.no_grad():
                logits = model.forward(prompt_batch)
                results.extend(logits[:, -1])
        else:
            steered_text = model.generate(prompt_batch, max_new_tokens=n, temperature=temp)
            results.extend(steered_text)

    model.reset_hooks()

    if return_layer_16:
        return torch.cat(results, dim=0)
    elif return_top_logits or return_all_logits:
        return results
    else:
        return list(map(tokenizer.decode, results))


def to_tensor_tokens(tokenizer: PreTrainedTokenizer, text: str) -> torch.Tensor:
    """Converts text to tensor tokens."""
    return tokenizer(text, return_tensors="pt")["input_ids"][0]


def project_onto_orthogonal_subspace(
    v: Float[torch.Tensor, D_MODEL], prev: Float[torch.Tensor, N * D_MODEL], R: float
) -> Float[torch.Tensor, D_MODEL]:
    """Projects vector v onto the orthogonal subspace of prev."""
    U = prev.t() / R
    return v - U @ U.t() @ v


def melbo_ortho(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    pretrained_theta: Float[torch.Tensor, D_MODEL],
    target_later: Float[torch.Tensor, D_MODEL],
    make_ortho_to: Float[torch.Tensor, N * D_MODEL],
    refine_epochs: int = 1000,
    enforce_orthogonality: bool = True,
) -> Float[torch.Tensor, D_MODEL]:
    """
    Implements MELBO orthogonalization algorithm.
    """
    dev = next(model.parameters()).device
    ortho_to_normalized = torch.nn.functional.normalize(make_ortho_to, dim=1)
    model.reset_hooks()
    pretrained_theta = pretrained_theta.to(dev)
    theta = nn.Parameter(pretrained_theta.clone().detach().to(dev), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.010)

    for epoch in (bar := trange(refine_epochs)):
        layer_16_acts = batch_steer_with_vec(
            model,
            tokenizer,
            theta[None, :],
            to_tensor_tokens(tokenizer, prompt),
            return_layer_16=True,
            progress_bar=False,
        )
        activation_difference_loss = (layer_16_acts.mean(dim=1) - target_later).norm()
        loss = activation_difference_loss
        opt.zero_grad()
        loss.backward()
        bar.set_postfix({"activation_difference": activation_difference_loss.item()})
        opt.step()

        if enforce_orthogonality:
            with torch.no_grad():
                theta.data = project_onto_orthogonal_subspace(theta.data, ortho_to_normalized, 1)

    return theta.clone().detach()
