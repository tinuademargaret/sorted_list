#%%
import os
import sys
import torch as t
from pathlib import Path

# Make sure exercises are in the path
# chapter = r"chapter1_transformer_interp"
# exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
# section_dir = exercises_dir / "monthly_algorithmic_problems" / "october23_sorted_list"
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from dataset import SortedListDataset
from model import create_model
from plotly_utils import hist, bar, imshow, line, scatter

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
#%
import einops
#%%
from eindex import eindex
from jaxtyping import Int, Float
#%%
from torch import Tensor
#%%
import functools
from tqdm import tqdm
from IPython.display import display
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv
# %%
dataset = SortedListDataset(size=1, list_len=5, max_value=10, seed=42)

print(dataset[0].tolist())
print(dataset.str_toks[0])

# %%
filename = "sorted_list_model.pt"

model = create_model(
    list_len=10,
    max_value=50,
    seed=0,
    d_model=96,
    d_head=48,
    n_layers=1,
    n_heads=2,
    normalization_type="LN",
    d_mlp=None
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);

# %%
W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))

W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))

W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))

b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))

W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))

W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))

b_V = model.b_V
t.testing.assert_close(b_V, t.zeros_like(b_V))

# %%
N = 500
dataset = SortedListDataset(size=N, list_len=10, max_value=50, seed=43)

logits, cache = model.run_with_cache(dataset.toks)
logits: t.Tensor = logits[:, dataset.list_len:-1, :]

targets = dataset.toks[:, dataset.list_len+1:]

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1)

batch_size, seq_len = dataset.toks.shape
logprobs_correct = eindex(logprobs, targets, "batch seq [batch seq]")
probs_correct = eindex(probs, targets, "batch seq [batch seq]")

avg_cross_entropy_loss = -logprobs_correct.mean().item()

print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")
# %%
print(model.cfg)
# %%
def show(dataset: SortedListDataset, batch_idx: int):

    logits: t.Tensor = model(dataset.toks)[:, dataset.list_len:-1, :]
    logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
    probs = logprobs.softmax(-1)

    str_targets = dataset.str_toks[batch_idx][dataset.list_len+1: dataset.seq_len]

    imshow(
        probs[batch_idx].T,
        y=dataset.vocab,
        x=[f"{dataset.str_toks[batch_idx][j]}<br><sub>({j})</sub>" for j in range(dataset.list_len+1, dataset.seq_len)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities:<br>Unsorted = ({','.join(dataset.str_toks[batch_idx][:dataset.list_len])})",
        text=[
            ["〇" if (str_tok == target) else "" for target in str_targets]
            for str_tok in dataset.vocab
        ],
        width=400,
        height=1000,
    )

show(dataset, 0)
# %%
"""
target = list_len+1 : seq_len
predictions = list_len: seq_len-1
model is causal
"""
print(model.cfg.attention_dir)
# %%
"""
Logit attribution

logits = W_U * ln(residiual)
residual = embed + ln(attn_out)
logits = (W_U * embed) + (W_U * attn_out) 
We would apply ln to the residual stack
"""
# Get W_U for answer tokens
answer_tokens = dataset.toks[:, dataset.list_len+1: dataset.seq_len]

logit_directions = model.tokens_to_residual_directions(answer_tokens)

assert(logit_directions.shape == t.Size([N, dataset.list_len, model.cfg.d_model]))

# %%
# function to project residual stream values to logit directions
def residual_stack_to_logit_dir(
        residual_stack: Float[Tensor, "... batch list_len d_model"],
        cache: ActivationCache,
        logit_directions: Float[Tensor, "batch list_len d_model"] = logit_directions

)-> Float[Tensor, "..."]:
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=(dataset.list_len, seq_len-1))

    average_logit = einops.einsum(
        scaled_residual_stack, logit_directions, 
        "... batch list_len d_model, batch list_len d_model -> ..."
    )/ N

    return average_logit

#%%
# Visualize accumulated logit 
accumulated_resid, labels = cache.accumulated_resid(layer=-1, pos_slice=(dataset.list_len, seq_len-1), return_labels=True)
assert(accumulated_resid.shape == t.Size([2, N, dataset.list_len, model.cfg.d_model]))

logit_lens_dir = residual_stack_to_logit_dir(accumulated_resid, cache)

assert(logit_lens_dir.shape == t.Size([2]))


line(
    logit_lens_dir, 
    hovermode="x unified",
    title="Logit Difference From Accumulated Residual Stream",
    labels={"x": "Layer", "y": "Logit Diff"},
    xaxis_tickvals=labels,
    width=800
)
# %%
# Layer attribution
per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=(dataset.list_len, seq_len-1), return_labels=True)
per_layer_logit_diffs = residual_stack_to_logit_dir(per_layer_residual, cache)

line(
    per_layer_logit_diffs, 
    hovermode="x unified",
    title="Logit Difference From Each Layer",
    labels={"x": "Layer", "y": "Logit Diff"},
    xaxis_tickvals=labels,
    width=800
)
# %%
# Head attribution
per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=(dataset.list_len, seq_len-1), return_labels=True)
per_head_residual = einops.rearrange(
    per_head_residual, 
    "(layer head) ... -> layer head ...", 
    layer=model.cfg.n_layers
)
per_head_logit_diffs = residual_stack_to_logit_dir(per_head_residual, cache)

imshow(
    per_head_logit_diffs, 
    labels={"x":"Head", "y":"Layer"}, 
    title="Logit Difference From Each Head",
    width=600
)
# %%
# Attention analysis
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=dataset.str_toks[0], attention=attention_pattern))
# %%
print( dataset.str_toks[0])

# %%
# %%
# Running model on numbers 1,2,3,4,5,6,7,8,9,10 (numbers that are next to each other)
sep_toks = t.full(size=(1,), fill_value=51)
unsorted_list = t.tensor([1, 3, 5, 4, 7, 9, 10, 2, 8, 6])
sorted_list = t.sort(unsorted_list, dim=-1).values
toks = t.concat([unsorted_list, sep_toks, sorted_list], dim=-1)
_, new_cache = model.run_with_cache(toks.to(device))

# %%
tok_str = [str(i) for i in toks.tolist()]
print(tok_str)
attention_pattern = cache["pattern", layer][:][0]
print(attention_pattern.shape)
display(cv.attention.attention_heads(tokens=tok_str, attention=attention_pattern))

# %%
