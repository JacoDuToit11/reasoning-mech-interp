import os
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from transformer_lens import patching
from jaxtyping import Float
from torch import Tensor

def delimiter_attention(model, prompts, delimiter=" because"):
    """
    Computes and plots the proportion of attention directed to a given delimiter token.
    Args:
        model: The language model with .to_tokens, .to_str_tokens, and .run_with_cache methods.
        prompts: List of strings to be analyzed.
        delimiter: Delimiter to focus attention on, e.g., ' because' or ' so'.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attn_prop_delim_all_prompts = t.zeros(n_layers, n_heads)

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        str_tokens = model.to_str_tokens(prompt)

        if delimiter not in str_tokens:
            print(f"Prompt '{prompt}' does not contain delimiter '{delimiter}'.")
            continue

        delim_idx = str_tokens.index(delimiter)
        for layer in range(n_layers):
            attn = cache["pattern", layer]
            delim_sum = attn[:, :, delim_idx].sum(dim=1)
            total_sum = attn.sum(dim=(1, 2))
            proportion = delim_sum / total_sum
            attn_prop_delim_all_prompts[layer] += proportion.cpu()

    attn_prop_delim_all_prompts /= len(prompts)

    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_prop_delim_all_prompts.numpy(), cmap="viridis")
    plt.title(f"Proportion of Attention to '{delimiter.strip()}'")
    plt.xlabel("Heads")
    plt.ylabel("Layers")
    plt.savefig(f"../results/{model_name}/attention_map_delim_{delimiter.strip()}.png")

def analyze_delimiter_attention(model):
    prompts = [
        'Alice went to the park because she wanted to find a treasure.',
        'Alice plays guitar because she enjoys strumming melodies.',
        'Alice went to Paris because Paris is a good place for art.'
    ]
    delimiter_attention(model, prompts, delimiter=" because")

    prompts = [
        'Bob and Chris got work to do so they are eager to explore.',
        'Bob and Chris made a cake so they are excited and happy.'
    ]
    delimiter_attention(model, prompts, delimiter=" so")

def cause_effect_attention(model, prompts, delimiter, direction):
    """
    Measures attention in a GPT-style model either from cause→effect or effect→cause.
    Args:
        model:      Your model with to_tokens, to_str_tokens, run_with_cache, etc.
        prompts:    List of strings to analyze.
        delimiter:  Token that separates cause from effect (e.g. " because", " so").
        direction:  "cause->effect" or "effect->cause".
    Returns:
        A (layers x heads) tensor of average attention proportions.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    attn_prop_all_prompts = t.zeros(n_layers, n_heads)

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        str_tokens = model.to_str_tokens(prompt)

        if delimiter not in str_tokens:
            continue

        delim_idx = str_tokens.index(delimiter)

        if direction == "effect->cause":
            cause_idxs = range(1, delim_idx)
            effect_idxs = range(delim_idx + 1, len(str_tokens))
        elif direction == "cause->effect":
            cause_idxs = range(delim_idx + 1, len(str_tokens))
            effect_idxs = range(1, delim_idx)
        else:
            raise ValueError("direction must be either 'cause->effect' or 'effect->cause'")

        for layer in range(n_layers):
            attn = cache["pattern", layer]

            if direction == "cause->effect":
                relevant_attn = attn[:, cause_idxs][:, :, effect_idxs]
            elif direction == "effect->cause":
                relevant_attn = attn[:, effect_idxs][:, :, cause_idxs]
            else:
                raise ValueError("direction must be either 'cause->effect' or 'effect->cause'")

            sum_relevant = relevant_attn.sum(dim=(1, 2)).cpu()
            total_sum = attn.sum(dim=(1, 2)).cpu()
            attn_prop_all_prompts[layer] += sum_relevant / total_sum

    attn_prop_all_prompts /= len(prompts)

    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_prop_all_prompts.numpy(), cmap="viridis")
    if direction == 'cause->effect':
        plt.title("Proportion of Cause-to-Effect Attention")
    else:
        plt.title("Proportion of Effect-to-Cause Attention")
    plt.xlabel("Heads")
    plt.ylabel("Layers")
    plt.savefig(f"../results/{model_name}/attention_map_cause_effect_{delimiter.strip()}.png")

def analyze_causal_attention(model):
    prompts = [
        "Alice went to the park because she wanted to find a treasure.",
        "Alice plays guitar because she enjoys strumming melodies.",
        "Alice went to Paris because Paris is a good place for art."
    ]

    cause_effect_attention(model, prompts,
                                        delimiter=" because",
                                        direction="effect->cause")

    prompts = [
        "Alice went to the craft fair so she could buy handmade gifts.",
        "Alice practiced daily so she would master the guitar."
    ]

    cause_effect_attention(model, prompts,
                                        delimiter=" so",
                                        direction="cause->effect")
    
def activation_patching(model, template,clean_tokens, corrupted_tokens, clean_answers, corr_answers):
    
    clean_answer_tokens = t.concat([
        model.to_tokens(answers, prepend_bos=False).T for answers in clean_answers
    ])
    corr_answer_tokens = t.concat([
        model.to_tokens(answers, prepend_bos=False).T for answers in clean_answers
    ])

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    def logits_to_ave_logit_diff(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch 2"],
        per_prompt: bool = False
        ) -> Float[Tensor, "*batch"]:
        '''
        Returns logit difference between the correct and incorrect answer.

        If per_prompt=True, return the array of differences rather than the average.
        '''

        final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
        answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
        correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
        answer_logit_diff = correct_logits - incorrect_logits
        return answer_logit_diff if per_prompt else answer_logit_diff.mean()

    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, clean_answer_tokens)
    print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, corr_answer_tokens)
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

    def logit_diff_metric(
        logits,
        answer_tokens=clean_answer_tokens,
        corrupted_logit_diff=corrupted_logit_diff,
        clean_logit_diff=clean_logit_diff,
    ):
        '''
        Linear function of logit diff, calibrated so that it equals 0 when performance is
        same as on corrupted input, and 1 when performance is same as on clean input.
        '''
        patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
        return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


    act_patch_resid_pre = patching.get_act_patch_resid_pre(
        model=model,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        patching_metric=logit_diff_metric
    )

    labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
    plt.figure(figsize=(8, 6))
    plt.imshow(
        act_patch_resid_pre.cpu(),
        cmap="viridis",
        aspect="auto"
    )

    plt.colorbar(label="Activation Values")
    plt.xlabel("Position")
    plt.ylabel("Layer")
    plt.title("Resid_Pre Activation Patching")

    if 'labels' in locals() and labels is not None:
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)

    plt.savefig(f"../results/{model_name}/activation_patching_per_block_{template}.png")

    act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
        model,
        corrupted_tokens,
        clean_cache,
        logit_diff_metric
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(act_patch_attn_head_out_all_pos.cpu(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Position")
    plt.title("Attention Head Outputs Heatmap")
    plt.savefig(f"../results/{model_name}/activation_patching_attn_head_out_all_pos_{template}.png")

def activation_patching_analysis(model):

    template = "ALB"
    clean_tokens = model.to_tokens(["John had to leave because he going to the."])
    corrupted_tokens = model.to_tokens(["John had to practice because he going to the."])

    clean_answers = [(" show")]
    corr_answers = [(" match")]

    activation_patching(model, template, clean_tokens, corrupted_tokens, clean_answers, corr_answers)

    # template = "AOB"
    # clean_tokens = model.to_tokens(["Jane will eat it because John is getting the", "Jane will eat it because John is getting the"])
    # corrupted_tokens = model.to_tokens(["Jane will sing it because John is getting the", "Jane will sing it because John is getting the"])

    # clean_answers = [(" sandwich", " music")]
    # corr_answers = [(" sandwich", " music")]
    # activation_patching(model, template, clean_tokens, corrupted_tokens, clean_answers, corr_answers)

def main():
    global model_name
    model_name = "gpt2-small"
    if not os.path.exists(f"../results/{model_name}"):
        os.makedirs(f"../results/{model_name}")
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
    analyze_delimiter_attention(model)
    analyze_causal_attention(model)
    activation_patching_analysis(model)

if __name__ == "__main__":
    main()
