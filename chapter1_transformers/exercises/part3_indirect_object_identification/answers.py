# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, scatter, bar
import part3_indirect_object_identification.tests as tests

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

MAIN = __name__ == "__main__"
# %%

if MAIN:
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )

# %%
# Here is where we test on a single prompt
# Result: 70% probability on Mary, as we expect

if MAIN:
    example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
    example_answer = " Mary"
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# %%
if MAIN:
    prompt_format = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    name_pairs = [
        (" Mary", " John"),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]

    # Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
    prompts = [
        prompt.format(name) 
        for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1] 
    ]
    # Define the answers for each prompt, in the form (correct, incorrect)
    answers = [names[::i] for names in name_pairs for i in (1, -1)]
    # Define the answer tokens (same shape as the answers)
    answer_tokens = t.concat([
        model.to_tokens(names, prepend_bos=False).T for names in answers
    ])

    rprint(prompts)
    rprint(answers)
    rprint(answer_tokens)
# %%
if MAIN:
    table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")

    for prompt, answer in zip(prompts, answers):
        table.add_row(prompt, repr(answer[0]), repr(answer[1]))

    rprint(table)

# %%
if MAIN:
    tokens = model.to_tokens(prompts, prepend_bos=True)
    # Move the tokens to the GPU
    tokens = tokens.to(device)
    # Run the model and cache all activations
    original_logits, cache = model.run_with_cache(tokens)

# %%
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    final_logits = logits[:, -1] # (batch d_vocab) #(batch)
    correct_logits = final_logits.gather(1, answer_tokens[:, 0].unsqueeze(1)).squeeze()
    incorrect_logits = final_logits.gather(1, answer_tokens[:, 1].unsqueeze(1)).squeeze()

    logit_diff = correct_logits - incorrect_logits # (batch)

    if per_prompt:
        return logit_diff

    return logit_diff.mean()
    


if MAIN:
    tests.test_logits_to_ave_logit_diff(logits_to_ave_logit_diff)

    original_per_prompt_diff = logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True)
    print("Per prompt logit difference:", original_per_prompt_diff)
    original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
    print("Average logit difference:", original_average_logit_diff)

    cols = [
        "Prompt", 
        Column("Correct", style="rgb(0,200,0) bold"), 
        Column("Incorrect", style="rgb(255,0,0) bold"), 
        Column("Logit Difference", style="bold")
    ]
    table = Table(*cols, title="Logit differences")

    for prompt, answer, logit_diff in zip(prompts, answers, original_per_prompt_diff):
        table.add_row(prompt, repr(answer[0]), repr(answer[1]), f"{logit_diff.item():.3f}")

    rprint(table)

"""
Part 2: Logit Attribution
"""

# %%
if MAIN:
    answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
    print("Answer residual directions shape:", answer_residual_directions.shape)

    correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions
    print(f"Logit difference directions shape:", logit_diff_directions.shape)
# %%
# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 

if MAIN:
    final_residual_stream: Float[Tensor, "batch seq d_model"] = cache["resid_post", -1]
    print(f"Final residual stream shape: {final_residual_stream.shape}")
    final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, -1, :]

    # Apply LayerNorm scaling (to just the final sequence position)
    # pos_slice is the subset of the positions we take - here the final token of each prompt
    scaled_final_token_residual_stream = cache.apply_ln_to_stack(final_token_residual_stream, layer=-1, pos_slice=-1)

    average_logit_diff = einops.einsum(
        scaled_final_token_residual_stream, logit_diff_directions,
        "batch d_model, batch d_model ->"
    ) / len(prompts)

    print(f"Calculated average logit diff: {average_logit_diff:.10f}")
    print(f"Original logit difference:     {original_average_logit_diff:.10f}")

    t.testing.assert_close(average_logit_diff, original_average_logit_diff)

# %%
def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    average_logit_diffs = einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / len(prompts)
    return average_logit_diffs


if MAIN:
    t.testing.assert_close(
        residual_stack_to_logit_diff(final_token_residual_stream, cache),
        original_average_logit_diff
    )

# %%
if MAIN:
    accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
    # accumulated_residual has shape (component, batch, d_model)

    logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(accumulated_residual, cache)

    line(
        logit_lens_logit_diffs, 
        hovermode="x unified",
        title="Logit Difference From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800
    )

# %%
if MAIN:
    per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)

    line(
        per_layer_logit_diffs, 
        hovermode="x unified",
        title="Logit Difference From Each Layer",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800
    )

# %%
if MAIN:
    per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    per_head_residual = einops.rearrange(
        per_head_residual, 
        "(layer head) ... -> layer head ...", 
        layer=model.cfg.n_layers
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)

    imshow(
        per_head_logit_diffs, 
        labels={"x":"Head", "y":"Layer"}, 
        title="Logit Difference From Each Head",
        width=600
    )

# %%
def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()



if MAIN:
    k = 3

    for head_type in ["Positive", "Negative"]:

        # Get the heads with largest (or smallest) contribution to the logit difference
        top_heads = topk_of_Nd_tensor(per_head_logit_diffs * (1 if head_type=="Positive" else -1), k)

        # Get all their attention patterns
        attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
            cache["pattern", layer][:, head][0]
            for layer, head in top_heads
        ])

        # Display results
        display(HTML(f"<h2>Top {k} {head_type} Logit Attribution Heads</h2>"))
        display(cv.attention.attention_patterns(
            attention = attn_patterns_for_important_heads,
            tokens = model.to_str_tokens(tokens[0]),
            attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
        ))
# %%
from transformer_lens import patching


# %%
clean_tokens = tokens
# Swap each adjacent pair to get corrupted tokens
indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(tokens))]
corrupted_tokens = clean_tokens[indices]

print(
    "Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
    "Corrupted string 0:", model.to_string(corrupted_tokens[0])
)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

# %%
def ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"], 
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    corrupted_logit_diff: float = corrupted_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is 
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


t.testing.assert_close(ioi_metric(clean_logits).item(), 1.0)
t.testing.assert_close(ioi_metric(corrupted_logits).item(), 0.0)
t.testing.assert_close(ioi_metric((clean_logits + corrupted_logits) / 2).item(), 0.5)

# %%
act_patch_resid_pre = patching.get_act_patch_resid_pre(
    model = model,
    corrupted_tokens = corrupted_tokens,
    clean_cache = clean_cache,
    patching_metric = ioi_metric
)

labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]

imshow(
    act_patch_resid_pre, 
    labels={"x": "Position", "y": "Layer"},
    x=labels,
    title="resid_pre Activation Patching",
    width=600
)
# %%
def patch_residual_component(
    corrupted_residual_component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    pos: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    '''
    corrupted_residual_component[:, pos] = clean_cache[hook.name][:, pos]
    return corrupted_residual_component

def get_act_patch_resid_pre(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], float]
) -> Float[Tensor, "layer pos"]:
    '''
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    out = t.empty((model.cfg.n_layers, corrupted_tokens.shape[1])).to(corrupted_tokens.device)
    for layer in range(model.cfg.n_layers):
        for pos in range(corrupted_tokens.shape[1]):
            model.reset_hooks()
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                return_type = "logits",
                fwd_hooks = [(utils.get_act_name("resid_pre", layer), partial(patch_residual_component, pos=pos, clean_cache=clean_cache))]
            )
            out[layer, pos] = patching_metric(patched_logits)
    return out





act_patch_resid_pre_own = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, ioi_metric)

t.testing.assert_close(act_patch_resid_pre, act_patch_resid_pre_own)
# %%
imshow(
    act_patch_resid_pre_own, 
    x=labels, 
    title="Logit Difference From Patched Residual Stream", 
    labels={"x":"Sequence Position", "y":"Layer"},
    width=600 # If you remove this argument, the plot will usually fill the available space
)

#%%
act_patch_block_every = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric)

imshow(
    act_patch_block_every,
    x=labels, 
    facet_col=0, # This argument tells plotly which dimension to split into separate plots
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"], # Subtitles of separate plots
    title="Logit Difference From Patched Attn Head Output", 
    labels={"x": "Sequence Position", "y": "Layer"},
    width=1000,
)

#%%
def get_act_patch_block_every(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], float]
) -> Float[Tensor, "3 layer pos"]:
    '''
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    out = t.empty((3, model.cfg.n_layers, corrupted_tokens.shape[1])).to(corrupted_tokens.device)
    for idx, block in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        for layer in range(model.cfg.n_layers):
            for pos in range(corrupted_tokens.shape[1]):
                model.reset_hooks()
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    return_type = "logits",
                    fwd_hooks = [(utils.get_act_name(block, layer), partial(patch_residual_component, pos=pos, clean_cache=clean_cache))]
                )
                out[idx, layer, pos] = patching_metric(patched_logits)
    return out

act_patch_block_every_own = get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric)

t.testing.assert_close(act_patch_block_every, act_patch_block_every_own)

imshow(
    act_patch_block_every_own,
    x=labels, 
    facet_col=0,
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
    title="Logit Difference From Patched Attn Head Output", 
    labels={"x": "Sequence Position", "y": "Layer"},
    width=1000
)

#%%
act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
    model, 
    corrupted_tokens, 
    clean_cache, 
    ioi_metric
)

imshow(
    act_patch_attn_head_out_all_pos, 
    labels={"y": "Layer", "x": "Head"}, 
    title="attn_head_out Activation Patching (All Pos)",
    width=600
)
# %%
def patch_head_vector(
    corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    '''
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector

def get_act_patch_attn_head_out_all_pos(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable
) -> Float[Tensor, "layer head"]:
    '''
    Returns an array of results of patching at all positions for each head in each
    layer, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    out = t.empty((model.cfg.n_layers, model.cfg.n_heads)).to(corrupted_tokens.device)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            model.reset_hooks()
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                return_type = "logits",
                fwd_hooks = [(utils.get_act_name("z", layer), partial(patch_head_vector, head_index=head, clean_cache=clean_cache))]
            )
            out[layer, head] = patching_metric(patched_logits)
    return out


act_patch_attn_head_out_all_pos_own = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, ioi_metric)

t.testing.assert_close(act_patch_attn_head_out_all_pos, act_patch_attn_head_out_all_pos_own)

imshow(
    act_patch_attn_head_out_all_pos_own,
    title="Logit Difference From Patched Attn Head Output", 
    labels={"x":"Head", "y":"Layer"},
    width=600
)

#%%
act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_all_pos_every(
    model, 
    corrupted_tokens, 
    clean_cache, 
    ioi_metric
)

imshow(
    act_patch_attn_head_all_pos_every, 
    facet_col=0, 
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Activation Patching Per Head (All Pos)", 
    labels={"x": "Head", "y": "Layer"},
)
# %%
# Get the heads with largest value patching
# (we know from plot above that these are the 4 heads in layers 7 & 8)
k = 4
top_heads = topk_of_Nd_tensor(act_patch_attn_head_all_pos_every[3], k=k)

# Get all their attention patterns
attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
    cache["pattern", layer][:, head].mean(0)
        for layer, head in top_heads
])

# Display results
display(HTML(f"<h2>Top {k} Logit Attribution Heads (from value-patching)</h2>"))
display(cv.attention.attention_patterns(
    attention = attn_patterns_for_important_heads,
    tokens = model.to_str_tokens(tokens[0]),
    attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
))

# %%
from part3_indirect_object_identification.ioi_dataset import NAMES, IOIDataset
N = 25
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(device)
)

# %%
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")


# %%
def format_prompt(sentence: str) -> str:
    '''Format a prompt by underlining names (for rich print)'''
    return re.sub("(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence) + "\n"


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
    '''Makes and displays a table, from cols rather than rows (using rich print)'''
    table = Table(*colnames, title=title)
    rows = list(zip(*cols))
    f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
    for row in rows[:n_rows]:
        table.add_row(*list(map(f, row)))
    rprint(table)

make_table(
    colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
    cols = [
        map(format_prompt, ioi_dataset.sentences), 
        model.to_string(ioi_dataset.s_tokenIDs).split(), 
        model.to_string(ioi_dataset.io_tokenIDs).split(), 
        map(format_prompt, abc_dataset.sentences), 
    ],
    title = "Sentences from IOI vs ABC distribution",
)

# %%
def logits_to_ave_logit_diff_2(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset = ioi_dataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()



model.reset_hooks(including_permanent=True)

ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

make_table(
    colnames = ["IOI prompt", "IOI logit diff", "ABC prompt", "ABC logit diff"],
    cols = [
        map(format_prompt, ioi_dataset.sentences), 
        ioi_per_prompt_diff,
        map(format_prompt, abc_dataset.sentences), 
        abc_per_prompt_diff,
    ],
    title = "Sentences from IOI vs ABC distribution",
)

# %%
def ioi_metric_2(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = ioi_average_logit_diff,
    corrupted_logit_diff: float = abc_average_logit_diff,
    ioi_dataset: IOIDataset = ioi_dataset,
) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset), 
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
    return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


print(f"IOI metric (IOI dataset): {ioi_metric_2(ioi_logits_original):.4f}")
print(f"IOI metric (ABC dataset): {ioi_metric_2(abc_logits_original):.4f}")
# %%

def patch_head_hook(old_head_vector:  Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int, 
    new_cache: ActivationCache
    ) -> Float[Tensor, "batch pos head_index d_head"]:
    
    old_head_vector[:, :, head_index] = new_cache[hook.name][:, :, head_index]
    return old_head_vector


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
) -> Float[Tensor, "layer head"]:
    
    out = t.empty(model.cfg.n_layers, model.cfg.n_heads)
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            model.reset_hooks()
            fwd_hooks = [(utils.get_act_name("z", layer), partial(patch_head_hook, head_index=head, new_cache=new_cache))]
            for freeze_layer in range(model.cfg.n_layers):
                for freeze_head in range(model.cfg.n_heads):
                    if freeze_layer == layer and freeze_head == head:
                        continue
                    fwd_hooks.append((utils.get_act_name("z", freeze_layer, freeze_head), partial(patch_head_hook, head_index=freeze_head, new_cache=orig_cache)))
            logits = model.run_with_hooks(orig_dataset.toks, fwd_hooks=fwd_hooks)
            out[layer, head] = patching_metric(logits)
    return out
            


path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, ioi_metric_2)

imshow(
    100 * path_patch_head_to_final_resid_post,
    title="Direct effect on logit difference",
    labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    width=600,
)

#%%

def patch_or_freeze_head_hook(old_head_vector:  Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int, 
    new_cache: ActivationCache,
    ) -> Float[Tensor, "batch pos head_index d_head"]:
    
    old_head_vector[:, :, head_index] = new_cache[hook.name][:, :, head_index]
    return old_head_vector

def receiver_cache_or_insert_hook(old_head_qkv_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_index: int,
    ) -> Float[Tensor, "batch pos head_index d_head"]:
    if hook.ctx.get(head_index, None) is None:
        hook.ctx[head_index] = old_head_qkv_vector[:, :, head_index]
    else:
        old_head_qkv_vector[:, :, head_index] = hook.ctx[head_index]
        del hook.ctx[head_index]
    return old_head_qkv_vector


def get_path_patch_head_to_heads(
    receiver_heads: List[Tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = None,
    orig_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "layer head"]:
    '''
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = input to a later head (or set of heads)

    The receiver node is specified by receiver_heads and receiver_input.
    Example (for S-inhibition path patching the queries):
        receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
        receiver_input = "v"

    Returns:
        tensor of metric values for every possible sender head
    '''
    # step 1 - cache all the activations
    model.reset_hooks()

    if new_cache is None:
        _, new_cache = model.run_with_cache(new_dataset.toks)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_dataset.toks)



    # step 2 - loop through all sender heads, caching the receiver inputs 

    out = t.empty(model.cfg.n_layers, model.cfg.n_heads)
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            model.reset_hooks()

            patching_hooks = []
            patching_hooks.append((utils.get_act_name("z", layer), partial(patch_or_freeze_head_hook, head_index=head, new_cache=new_cache)))

            freezing_hooks = []
            for freeze_layer in range(model.cfg.n_layers):
                for freeze_head in range(model.cfg.n_heads):
                    if (freeze_layer == layer) and (freeze_head == head):
                        continue
                    freezing_hooks.append((utils.get_act_name("z", freeze_layer), partial(patch_or_freeze_head_hook, head_index=freeze_head, new_cache=orig_cache)))

            receiver_hooks = []
            for receiver_head in receiver_heads:
                receiver_layer_idx, receiver_head_idx = receiver_head
                receiver_hooks.append((utils.get_act_name(receiver_input, receiver_layer_idx), partial(receiver_cache_or_insert_hook, head_index=receiver_head_idx)))

            _ = model.run_with_hooks(orig_dataset.toks, fwd_hooks=patching_hooks + freezing_hooks + receiver_hooks)

    # step 3 - run the model again, with only the receiver hooks, on the clean inputs

            logits = model.run_with_hooks(orig_dataset.toks, fwd_hooks=receiver_hooks)
            out[layer, head] = patching_metric(logits)
    return out

model.reset_hooks()

s_inhibition_value_path_patching_results = get_path_patch_head_to_heads(
    receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
    receiver_input = "v",
    model = model,
    patching_metric = ioi_metric_2
)

imshow(
    100 * s_inhibition_value_path_patching_results,
    title="Direct effect on S-Inhibition Heads' values", 
    labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
    width=600,
    coloraxis=dict(colorbar_ticksuffix = "%"),
)

# %%
def scatter_embedding_vs_attn(
    attn_from_end_to_io: Float[Tensor, "batch"],
    attn_from_end_to_s: Float[Tensor, "batch"],
    projection_in_io_dir: Float[Tensor, "batch"],
    projection_in_s_dir: Float[Tensor, "batch"],
    layer: int,
    head: int
):
    scatter(
        x=t.concat([attn_from_end_to_io, attn_from_end_to_s], dim=0),
        y=t.concat([projection_in_io_dir, projection_in_s_dir], dim=0),
        color=["IO"] * N + ["S"] * N,
        title=f"Projection of the output of {layer}.{head} along the name<br>embedding vs attention probability on name",
        title_x=0.5,
        labels={"x": "Attn prob on name", "y": "Dot w Name Embed", "color": "Name type"},
        color_discrete_sequence=["#72FF64", "#C9A5F7"],
        width=650
    )
# %%
def calculate_and_show_scatter_embedding_vs_attn(
    layer: int,
    head: int,
    cache: ActivationCache = ioi_cache,
    dataset: IOIDataset = ioi_dataset,
) -> None:
    '''
    Creates and plots a figure equivalent to 3(c) in the paper.

    This should involve computing the four 1D tensors:
        attn_from_end_to_io
        attn_from_end_to_s
        projection_in_io_dir
        projection_in_s_dir
    and then calling the scatter_embedding_vs_attn function.
    '''
    attn = cache[utils.get_act_name("pattern", layer)][:, head]
    ends = dataset.word_idx["end"]
    IOs = dataset.word_idx["IO"]
    Ss = dataset.word_idx["S1"]
    batch_size = attn.shape[0]
    attn_from_end_to_io = attn[range(0, batch_size), ends, IOs]
    attn_from_end_to_s = attn[range(0, batch_size), ends, Ss]

    unembeds: Float[Tensor, "d_model d_vocab"] = model.W_U.T
    z: Float[Tensor, "batch seq d_head"] = cache[utils.get_act_name("z", layer)][:, :, head]
    W_O: Float[Tensor, "d_head d_model"] = model.W_O[layer, head]
    head_out: Float[Tensor, "batch seq d_model"] = z @ W_O

    head_out_end: Float[Tensor, "batch d_model"] = head_out[range(0, batch_size), ends]
    IO_embeds: Float[Tensor, "batch d_model"] = unembeds[dataset.io_tokenIDs]
    S_embeds: Float[Tensor, "batch d_model"] = unembeds[dataset.s_tokenIDs]

    projection_in_io_dir = einops.einsum(head_out_end, IO_embeds, "batch d_model, batch d_model -> batch")
    projection_in_s_dir = einops.einsum(head_out_end, S_embeds, "batch d_model, batch d_model -> batch")

    scatter_embedding_vs_attn(
         attn_from_end_to_io=attn_from_end_to_io,
         attn_from_end_to_s=attn_from_end_to_s,
         projection_in_io_dir=projection_in_io_dir,
         projection_in_s_dir=projection_in_s_dir,
         layer=layer,
         head=head
     )



nmh = (9, 9)
calculate_and_show_scatter_embedding_vs_attn(*nmh)

nnmh = (11, 10)
calculate_and_show_scatter_embedding_vs_attn(*nnmh)

#%%
def get_copying_scores(
    model: HookedTransformer,
    k: int = 5,
    names: list = NAMES
) -> Float[Tensor, "2 layer-1 head"]:
    '''
    Gets copying scores (both positive and negative) as described in page 6 of the IOI paper, for every (layer, head) pair in the model.

    Returns these in a 3D tensor (the first dimension is for positive vs negative).

    Omits the 0th layer, because this is before MLP0 (which we're claiming acts as an extended embedding).
    '''
    name_tokens = model.to_tokens(names, prepend_bos=False)
    name_embeds: Float[Tensor, "batch 1 d_model"] = model.W_E[name_tokens]
    name_mlp: Float[Tensor, "batch 1 d_model"]= model.blocks[0].mlp(model.blocks[0].ln2(name_embeds))
    
    out = t.empty(2, model.cfg.n_layers - 1, model.cfg.n_heads)
    for layer in range(1, model.cfg.n_layers):
        for head in range(model.cfg.n_heads):

            W_O = model.W_O[layer, head]
            W_V = model.W_V[layer, head]
            W_OV = W_V @ W_O
            
            name_mlp_out_pos = model.ln_final(name_mlp @ W_OV) @ model.W_U
            name_mlp_out_pos = name_mlp_out_pos.squeeze(1)

            top_k_pos = t.topk(name_mlp_out_pos, k=k, dim=1).indices
            positive_copying_score = (top_k_pos == name_tokens).any(dim=1).float().mean()

            name_mlp_out_neg = model.ln_final(-name_mlp @ W_OV) @ model.W_U
            name_mlp_out_neg = name_mlp_out_neg.squeeze(1)

            top_k_neg = t.topk(name_mlp_out_neg, k=k, dim=1).indices
            negative_copying_score = (top_k_neg == name_tokens).any(dim=1).float().mean()

            out[0, layer-1, head] = positive_copying_score
            out[1, layer-1, head] = negative_copying_score

    return out
            




copying_results = get_copying_scores(model)

imshow(
    copying_results, 
    facet_col=0, 
    facet_labels=["Positive copying scores", "Negative copying scores"],
    title="Copying scores of attention heads' OV circuits",
    width=800
)


heads = {"name mover": [(9, 9), (10, 0), (9, 6)], "negative name mover": [(10, 7), (11, 10)]}

for i, name in enumerate(["name mover", "negative name mover"]):
    make_table(
        title=f"Copying Scores ({name} heads)",
        colnames=["Head", "Score"],
        cols=[
            list(map(str, heads[name])) + ["[dark_orange bold]Average"],
            [f"{copying_results[i, layer-1, head]:.2%}" for (layer, head) in heads[name]] + [f"[dark_orange bold]{copying_results[i].mean():.2%}"]
        ]
    )

# %%
t.cuda.empty_cache()

def get_attn_scores(
    model: HookedTransformer, 
    seq_len: int, 
    batch: int, 
    head_type: Literal["duplicate", "prev", "induction"]
):
    '''
    Returns attention scores for sequence of duplicated tokens, for every head.
    '''
    toks = t.randint(0, model.cfg.d_vocab, (batch, seq_len))
    rep_toks = t.cat([toks, toks], dim=1)

    head_type_to_offset_dict = {
        "duplicate": seq_len,
        "prev": 1,
        "induction": seq_len - 1
    }

    offset = head_type_to_offset_dict[head_type]

    out = t.empty(model.cfg.n_layers, model.cfg.n_heads)

    logits, cache = model.run_with_cache(rep_toks, names_filter = lambda name: name.endswith("pattern"))

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn_scores = cache["pattern", layer][:, head]
            out[layer, head] = attn_scores.diagonal(offset=offset, dim1=-1, dim2=-2).mean()
    
    return out


def plot_early_head_validation_results(seq_len: int = 50, batch: int = 50):
    '''
    Produces a plot that looks like Figure 18 in the paper.
    '''
    head_types = ["duplicate", "prev", "induction"]

    results = t.stack([
        get_attn_scores(model, seq_len, batch, head_type=head_type)
        for head_type in head_types
    ])

    imshow(
        results,
        facet_col=0,
        facet_labels=[
            f"{head_type.capitalize()} token attention prob.<br>on sequences of random tokens"
            for head_type in head_types
        ],
        labels={"x": "Head", "y": "Layer"},
        width=1300,
    )



model.reset_hooks()
plot_early_head_validation_results()

# %%