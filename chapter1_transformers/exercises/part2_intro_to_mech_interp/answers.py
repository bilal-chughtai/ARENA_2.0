# %%
# autoreload
%load_ext autoreload
%autoreload 2
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

"""
1: TransformerLens: Introduction
"""
# %%
if MAIN:
    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")


# %%
print(gpt2_small.cfg)

# %%
if MAIN:
    model_description_text = '''## Loading Models

    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

    loss = gpt2_small(model_description_text, return_type="loss")
    print("Model loss:", loss)

# %%
if MAIN:
    print(gpt2_small.to_str_tokens("gpt2"))
    print(gpt2_small.to_tokens(["gpt2", "hello"]))
    print(gpt2_small.to_string([50256, 70, 457, 17]))
    print(gpt2_small.to_string([50256, 31373, 50256, 50256]))

# %%
if MAIN:
    logits: Tensor = gpt2_small(model_description_text, return_type="logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]
    # YOUR CODE HERE - get the model's prediction on the text
    true_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
    num_correct = (prediction == true_tokens).sum()

    print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
    print(f"Correct words: {gpt2_small.to_str_tokens(prediction[prediction == true_tokens])}")

# %%
if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# %%
if MAIN:
    attn_patterns_layer_0 = gpt2_cache["pattern", 0]    
if MAIN:
    attn_patterns_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]
    t.testing.assert_close(attn_patterns_layer_0, attn_patterns_layer_0_copy)

# %%
if MAIN:
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]

    # YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)

    layer0_pattern_from_q_and_k = None
    q = gpt2_cache["q", 0]
    k = gpt2_cache["k", 0]
    print(q.shape)
    print(k.shape)
    pos = q.shape[0]

    attn = einops.einsum(q, k, "posq n_heads d_head, posk n_heads d_head -> n_heads posq posk ") / np.sqrt(gpt2_small.cfg.d_head)
    mask = t.tril(t.ones(pos, pos), diagonal=-1).T.bool().to(device)
    masked_attn = attn.masked_fill(mask, -t.inf)
    attn_weights = F.softmax(masked_attn, dim=-1)
    layer0_pattern_from_q_and_k = attn_weights

    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    print("Tests passed!")

# %%
if MAIN:
    print(type(gpt2_cache))
    attention_pattern = gpt2_cache["pattern", 0, "attn"]
    print(attention_pattern.shape)
    gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

    print("Layer 0 Head Attention Patterns:")
    display(cv.attention.attention_patterns(
        tokens=gpt2_str_tokens, 
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    ))

"""
2: Finding Induction Heads
"""
# %%
if MAIN:
    cfg = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True, # defaults to False
        tokenizer_name="EleutherAI/gpt-neox-20b", 
        seed=398,
        use_attn_result=True,
        normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
        positional_embedding_type="shortformer"
    )

# %%
if MAIN:
    weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()

    if not weights_dir.exists():
        url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
        output = str(weights_dir)
        gdown.download(url, output)

# %%
if MAIN:
    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_dir, map_location=device)
    model.load_state_dict(pretrained_weights)

# %%
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

    logits, cache = model.run_with_cache(text, remove_batch_dim=True)


    print("Layer 0 Head Attention Patterns:")
    layer_0_attention = cache["pattern", 0, "attn"]
    display(cv.attention.attention_patterns(
        tokens=model.to_str_tokens(text),
        attention=layer_0_attention,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    ))

    layer_1_attention = cache["pattern", 1, "attn"]
    print("Layer 1 Head Attention Patterns:")
    display(cv.attention.attention_patterns(
        tokens=model.to_str_tokens(text),
        attention = layer_1_attention,
        attention_head_names=[f"L1H{i}" for i in range(12)],
    ))
# %%

def get_all_attn_patterns(cache: ActivationCache) -> Tuple[str, t.Tensor]:
    '''
    Returns a dictionary of the form
    {
        "layer.head": attn_pattern
    }
    where attn_pattern is a tensor of shape (pos, pos) representing the attention pattern for the head "layer.head"
    '''
    for layer in range(2):
        for head in range(12):
            yield f"{layer}.{head}", cache["pattern", layer, "attn"][head]


def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    attn_patterns = get_all_attn_patterns(cache)
    current_head_detectors = []
    for layer_head, attn_pattern in attn_patterns:
        score = attn_pattern.diag().mean()
        if score > 0.2:
            current_head_detectors.append(layer_head)
    return current_head_detectors



def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    attn_patterns = get_all_attn_patterns(cache)
    prev_head_detectors = []
    for layer_head, attn_pattern in attn_patterns:
        score = attn_pattern.diag(-1).mean()
        if score > 0.8:
            prev_head_detectors.append(layer_head)
    return prev_head_detectors

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    attn_patterns = get_all_attn_patterns(cache)
    first_head_detectors = []
    for layer_head, attn_pattern in attn_patterns:
        score = attn_pattern[:, 0].mean()
        if score > 0.8:
            first_head_detectors.append(layer_head)
    return first_head_detectors


if MAIN:
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    random_tokens = t.randint(0, model.cfg.d_vocab, (batch, seq_len)).to(device)
    start_token = t.tensor([model.tokenizer.bos_token_id] * batch).unsqueeze(-1).to(device)
    rep_tokens = t.cat([start_token, random_tokens, random_tokens], dim=-1)
    return rep_tokens

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens, remove_batch_dim=True)
    return rep_tokens, rep_logits, rep_cache


if MAIN:
    seq_len = 50
    batch = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_cache.remove_batch_dim()
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    plot_loss_difference(log_probs, rep_str, seq_len)

#%%
layer_1_attention = rep_cache["pattern", 1, "attn"]
print("Layer 1 Head Attention Patterns:")
display(cv.attention.attention_patterns(
    tokens=rep_str,
    attention = layer_1_attention,
    attention_head_names=[f"L1H{i}" for i in range(12)],
))

# %%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    attn_patterns = get_all_attn_patterns(cache)
    induction_head_detectors = []
    seq_len = (cache["pattern", 0, "attn"][0].shape[0]-1) // 2
    print(seq_len)
    for layer_head, attn_pattern in attn_patterns:
        score = attn_pattern.diag(-seq_len+1).mean()
        print(layer_head, score)
        if score > 0.4:
            induction_head_detectors.append(layer_head)
    return induction_head_detectors


if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

"""
Part 3: TransformerLens -- Hooks
"""
# %%
if MAIN:
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

    # We make a tensor to store the induction score for each head.
    # We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    scores = einops.reduce(t.diagonal(pattern, offset=-seq_len+1, dim1=-2, dim2=-1), "batch head_index pos -> head_index", "mean")
    layer_index = int(hook.name.split(".")[1])
    induction_score_store[layer_index] = scores



if MAIN:
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    model.run_with_hooks(
        rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            induction_score_hook
        )]
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store, 
        labels={"x": "Head", "y": "Layer"}, 
        title="Induction Score by Head", 
        text_auto=".2f",
        width=900, height=400
    )

# %%
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
            attention=pattern.mean(0)
        )
    )


if MAIN:
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(gpt2_small, seq_len, batch)

    # We make a tensor to store the induction score for each head.
    # We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)

if MAIN:
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    gpt2_small.run_with_hooks(
        rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            induction_score_hook
        ),
        (
            pattern_hook_names_filter,
            visualize_pattern_hook
        )]
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store, 
        labels={"x": "Head", "y": "Layer"}, 
        title="Induction Score by Head", 
        text_auto=".2f",
        width=900, height=400
    )

    
# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Float[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]] # (d_model, seq-1)
    print(W_U_correct_tokens.shape)
    
    embed_attribution = einops.einsum(embed[:-1], W_U_correct_tokens, "seq1 d_model, d_model seq2 -> seq1 seq2").diag().unsqueeze(-1)
    print(embed_attribution.shape)

    l1_attribution = einops.einsum(l1_results[:-1], W_U_correct_tokens, "seq1 nheads d_model, d_model seq2 -> seq1 seq2 nheads").diagonal(dim1=0, dim2=1).T
    l2_attribution = einops.einsum(l2_results[:-1], W_U_correct_tokens, "seq1 nheads d_model, d_model seq2 -> seq1 seq2 nheads").diagonal(dim1=0, dim2=1).T
    print(l1_attribution.shape, l2_attribution.shape) 
    return t.cat([embed_attribution, l1_attribution, l2_attribution], dim=-1)

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)

    with t.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result", 0]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
        print("Tests passed!")

#%%
if MAIN:
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

    plot_logit_attribution(model, logit_attr, tokens)

# %%
if MAIN:
    seq_len = 50

    embed = rep_cache["embed"]
    l1_results = rep_cache["result", 0]
    l2_results = rep_cache["result", 1]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]

    # YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
    first_half_logit_attr = logit_attribution(embed[:1+seq_len], l1_results[:1+seq_len], l2_results[:1+seq_len], model.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)


    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

    plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
    plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")

# %%
def head_ablation_hook(
    attn_result: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_model"]:
    '''
    A hook function that zeros out the activations of a single head.
    '''
    attn_result[:, :, head_index_to_ablate] = 0
    return attn_result

def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()



def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("result", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores



if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)

if MAIN:
    imshow(
        ablation_scores, 
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Logit Difference After Ablating Heads", 
        text_auto=".2f",
        width=900, height=400
    )

# %% ablate every head except 0.7, 1.4 and 1.10

def ablate_all_non_important_heads(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    
    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    temp_hooks = []
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            if (layer, head) not in [(0, 7), (1, 4), (1, 10)]:
            # Use functools.partial to create a temporary hook function with the head number fixed
                temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
                temp_hooks.append((utils.get_act_name("result", layer), temp_hook_fn))
            # Run the model with the ablation hook
    ablated_logits = model.run_with_hooks(tokens, fwd_hooks=temp_hooks)
    # Calculate the logit difference
    loss = cross_entropy_loss(ablated_logits, tokens)
    # Store the result, subtracting the clean loss so that a value of zero means no change in loss
    ablation_score = loss - loss_no_ablation

    return ablation_score

if MAIN:
    ablation_score = ablate_all_non_important_heads(model, rep_tokens)
    print("Ablation score ablaiting all non important heads: ", ablation_score)


# %% mean ablate instead 

def mean_ablation_hook(
    attn_result: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_model"]:
    '''
    A hook function that zeros out the activations of a single head.
    '''
    attn_result[:, :, head_index_to_ablate] = attn_result[:, :, head_index_to_ablate].mean(dim=-1, keepdim=True)
    return attn_result

def get_mean_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(mean_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("result", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores



if MAIN:
    ablation_scores = get_mean_ablation_scores(model, rep_tokens)

if MAIN:
    imshow(
        ablation_scores, 
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Logit Difference After Ablating Heads", 
        text_auto=".2f",
        width=900, height=400
    )

"""
Part 4: Reverse Engineering Induction Circuits
"""
# %%
if MAIN:
    A = t.randn(5, 2)
    B = t.randn(2, 5)
    AB = A @ B
    AB_factor = FactoredMatrix(A, B)
    print("Norms:")
    print(AB.norm())
    print(AB_factor.norm())

    print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")
# %%
if MAIN:
    print("Eigenvalues:")
    print(t.linalg.eig(AB).eigenvalues)
    print(AB_factor.eigenvalues)
    print()
    print("Singular Values:")
    print(t.linalg.svd(AB).S)
    print(AB_factor.S)
    print("Full SVD:")
    print(AB_factor.svd())
    

# %%
if MAIN:
    C = t.randn(5, 300)
    ABC = AB @ C
    ABC_factor = AB_factor @ C
    print("Unfactored:", ABC.shape, ABC.norm())
    print("Factored:", ABC_factor.shape, ABC_factor.norm())
    print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")
if MAIN:
    AB_unfactored = AB_factor.AB
    t.testing.assert_close(AB_unfactored, AB)
# %%
if MAIN:
    head_index = 4
    layer = 1

    W_O = model.W_O[layer, head_index]
    W_V = model.W_V[layer, head_index]
    W_E = model.W_E
    W_U = model.W_U

    OV_circuit = FactoredMatrix(W_V, W_O)
    full_OV_circuit = W_E @ OV_circuit @ W_U

    tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)
# %%
if MAIN:
    randints = t.randint(0, model.cfg.d_vocab, (200, ))
    full_OV_circuit_sample = full_OV_circuit[randints, randints]
    full_OV_circuit_sample = full_OV_circuit_sample.AB

    imshow(
        full_OV_circuit_sample,
        labels={"x": "Input token", "y": "Logits on output token"},
        title="Full OV circuit for copying head",
        width=700,
    )

# %%
def top_1_acc(full_OV_circuit: FactoredMatrix) -> float:
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    correct = 0
    for i in range(full_OV_circuit.ldim):
        row_i = full_OV_circuit.A[i, :] @ full_OV_circuit.B
        if row_i.argmax().item() == i:
            correct += 1
    return correct / full_OV_circuit.ldim

def top_5_acc(full_OV_circuit: FactoredMatrix) -> float:
    correct = 0
    for i in range(full_OV_circuit.ldim):
        row_i = full_OV_circuit.A[i, :] @ full_OV_circuit.B
        top_5 = row_i.topk(5).indices
        if i in top_5:
            correct += 1
    return correct / full_OV_circuit.ldim

if MAIN:
    print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")
    print(f"Fraction of the time that the correct logit is in the top 5: {top_5_acc(full_OV_circuit):.4f}")

# %%
if MAIN:
    W_O_both = einops.rearrange(model.W_O[1, [4, 10]], "head d_head d_model -> (head d_head) d_model")
    W_V_both = einops.rearrange(model.W_V[1, [4, 10]], "head d_model d_head -> d_model (head d_head) ")
    W_E = model.W_E
    W_U = model.W_U

    OV_circuit = FactoredMatrix(W_V_both, W_O_both)
    effective_OV_circuit = W_E @ OV_circuit @ W_U

    print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(effective_OV_circuit):.4f}")
    print(f"Fraction of the time that the correct logit is in the top 5: {top_5_acc(effective_OV_circuit):.4f}") 

# %%
def mask_scores(attn_scores: Float[Tensor, "query_nctx key_nctx"]):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    assert attn_scores.shape == (model.cfg.n_ctx, model.cfg.n_ctx)
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores



if MAIN:
    # YOUR CODE HERE - calculate the matrix `pos_by_pos_pattern` as described above
    W_pos = model.W_pos
    print(W_pos.shape)
    W_Q = model.W_Q[0, 7]
    print(W_Q.shape)
    W_K = model.W_K[0, 7]
    print(W_K.shape)

    QK_circuit = W_Q @ W_K.T
    full_qk_circuit = W_pos @ QK_circuit @ W_pos.T / t.sqrt(t.tensor(model.cfg.d_head).to(W_pos.device))
    print(full_qk_circuit.shape)
    pos_by_pos_pattern = mask_scores(full_qk_circuit) 
    print(pos_by_pos_pattern.shape)
    pos_by_pos_pattern = pos_by_pos_pattern.softmax(dim=-1)
    
    print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")

    imshow(
        utils.to_numpy(pos_by_pos_pattern[:100, :100]), 
        labels={"x": "Key", "y": "Query"}, 
        title="Attention patterns for prev-token QK circuit, first 100 indices",
        width=700
    )

    tests.test_pos_by_pos_pattern(pos_by_pos_pattern, model, layer, head_index)

# %%
def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, 0, 0]th element is y_i (from notation above)
    '''
    e = cache["embed"].unsqueeze(0)
    pe = cache["pos_embed"].unsqueeze(0)
    x = einops.rearrange(cache["result", 0], "seq n_heads d_model -> n_heads seq d_model")
    return t.cat([e, pe, x], dim=0)

def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]

    The [i, 0, 0]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    '''
    W_Q = model.W_Q[1, ind_head_index]
    print(W_Q.shape)
    print(decomposed_qk_input.shape)
    return einops.einsum(decomposed_qk_input, W_Q, "n_heads seq d_model, d_model d_head -> n_heads seq d_head")

def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]

    The [i, 0, 0]th element is y_i @ W_K(so the sum along axis 0 is just the k-values)
    '''
    W_K = model.W_K[1, ind_head_index]
    return einops.einsum(decomposed_qk_input, W_K, "n_heads seq d_model, d_model d_head -> n_heads seq d_head")


if MAIN:
    ind_head_index = 4
    # First we get decomposed q and k input, and check they're what we expect
    decomposed_qk_input = decompose_qk_input(rep_cache)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)
    # Second, we plot our results
    component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
    for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
        imshow(
            utils.to_numpy(decomposed_input.pow(2).sum([-1])), 
            labels={"x": "Position", "y": "Component"},
            title=f"Norms of components of {name}", 
            y=component_labels,
            width=1000, height=400
        )

# %%
def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    '''
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    '''
    return einops.einsum(decomposed_q, decomposed_k, "n_headsq seqq d_head, n_headsk seqk d_head -> n_headsq n_headsk seqq seqk")


if MAIN:
    tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k)

if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = einops.reduce(
        decomposed_scores, 
        "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", 
        t.std
    )

    # First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
    imshow(
        utils.to_numpy(t.tril(decomposed_scores[0, 9])), 
        title="Attention score contributions from (query, key) = (embed, output of L0H7)",
        width=800
    )

    # Second plot: std dev over query and key positions, shown by component
    imshow(
        utils.to_numpy(decomposed_stds), 
        labels={"x": "Key Component", "y": "Query Component"},
        title="Standard deviations of attention score contributions (by key and query component)", 
        x=component_labels, 
        y=component_labels,
        width=800
    )
# %%
def find_K_comp_full_circuit(
    model: HookedTransformer,
    prev_token_head_index: int,
    ind_head_index: int
) -> FactoredMatrix:
    '''
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    W_E = model.W_E
    W_Q = model.W_Q[1, ind_head_index]
    W_K = model.W_K[1, ind_head_index]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]

    W_QK = FactoredMatrix(W_Q, W_K.T)
    W_OV = FactoredMatrix(W_V, W_O)

    return W_E @ W_QK @ W_OV.T @ W_E.T



if MAIN:
    prev_token_head_index = 7
    ind_head_index = 4
    K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

    print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}")

# %%
def get_comp_score(
    W_A: Float[Tensor, "in_A out_A"], 
    W_B: Float[Tensor, "out_A out_B"]
) -> float:
    '''
    Return the composition score between W_A and W_B.
    '''
    W_AB = W_A @ W_B
    return t.sqrt(W_AB.pow(2).sum() / (W_A.pow(2).sum() * W_B.pow(2).sum())).item()


if MAIN:
    tests.test_get_comp_score(get_comp_score)

# Get all QK and OV matrices

if MAIN:
    W_QK = model.W_Q @ model.W_K.transpose(-1, -2)
    W_OV = model.W_V @ model.W_O

    # Define tensors to hold the composition scores
    composition_scores = {
        "Q": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
        "K": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
        "V": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
    }


    for i in range(model.cfg.n_heads):
        for j in range(model.cfg.n_heads):
            composition_scores["Q"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j])
            composition_scores["K"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j].T)
            composition_scores["V"][i, j] = get_comp_score(W_OV[0, i], W_OV[1, j])

    for comp_type in "QKV":
        plot_comp_scores(model, composition_scores[comp_type], f"{comp_type} Composition Scores").show()
# %%
def generate_single_random_comp_score() -> float:
    '''
    Write a function which generates a single composition score for random matrices
    '''
    W_Q = t.empty(model.cfg.d_head, model.cfg.d_model)
    W_K = t.empty(model.cfg.d_head, model.cfg.d_model)
    W_V = t.empty(model.cfg.d_head, model.cfg.d_model)
    W_O = t.empty(model.cfg.d_head, model.cfg.d_model)
    nn.init.kaiming_uniform_(W_Q, a=np.sqrt(5))
    nn.init.kaiming_uniform_(W_K, a=np.sqrt(5))
    nn.init.kaiming_uniform_(W_V, a=np.sqrt(5))
    nn.init.kaiming_uniform_(W_O, a=np.sqrt(5))

    W_QK = W_Q @ W_K.T
    W_OV = W_V @ W_O.T

    return get_comp_score(W_QK, W_OV)

if MAIN:
    n_samples = 300
    comp_scores_baseline = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        comp_scores_baseline[i] = generate_single_random_comp_score()
    print("\nMean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    hist(
        comp_scores_baseline, 
        nbins=50, 
        width=800, 
        labels={"x": "Composition score"}, 
        title="Random composition scores"
    )

if MAIN:
    baseline = comp_scores_baseline.mean()
    for comp_type, comp_scores in composition_scores.items():
        plot_comp_scores(model, comp_scores, f"{comp_type} Composition Scores", baseline=baseline).show()

# %%
def ablation_induction_score(prev_head_index: Optional[int], ind_head_index: int) -> float:
    '''
    Takes as input the index of the L0 head and the index of the L1 head, and then runs with the previous token head ablated and returns the induction score for the ind_head_index now.
    '''

    def ablation_hook(v, hook):
        if prev_head_index is not None:
            v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx[prev_head_index] = attn[0, ind_head_index].diag(-(seq_len - 1)).mean()

    model.run_with_hooks(
        rep_tokens,
        fwd_hooks=[
            (utils.get_act_name("v", 0), ablation_hook),
            (utils.get_act_name("pattern", 1), induction_pattern_hook)
        ],
    )
    return model.blocks[1].attn.hook_pattern.ctx[prev_head_index].item()



if MAIN:
    baseline_induction_score = ablation_induction_score(None, 4)
    print(f"Induction score for no ablations: {baseline_induction_score:.5f}\n")
    for i in range(model.cfg.n_heads):
        new_induction_score = ablation_induction_score(i, 4)
        induction_score_change = new_induction_score - baseline_induction_score
        print(f"Ablation score change for head {i:02}: {induction_score_change:+.5f}")
# %%
