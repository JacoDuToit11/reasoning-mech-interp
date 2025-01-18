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
    
    This function analyzes how much attention each head in the transformer pays to delimiter
    words like 'because' or 'so', creating a heatmap visualization of the attention patterns.
    
    Args:
        model: The language model
        prompts: List of strings to be analyzed
        delimiter: Delimiter to focus attention on, e.g., ' because' or ' so'
    """
    print(f"\nAnalyzing attention patterns for delimiter: '{delimiter}'")
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
    """
    Runs delimiter attention analysis on two sets of prompts:
    1. Prompts containing 'because'
    2. Prompts containing 'so'
    
    Creates heatmaps showing how different attention heads focus on these words.
    
    Args:
        model: The transformer model to analyze
    """
    print("Starting delimiter attention analysis...")
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
    Measures attention patterns between cause and effect parts of sentences.
    
    This function analyzes how attention flows between the cause and effect portions
    of sentences, separated by delimiters like 'because' or 'so'. It can analyze
    attention in both directions.
    
    Args:
        model: model
        prompts: List of strings to analyze
        delimiter: Token that separates cause from effect (e.g. " because", " so")
        direction: "cause->effect" or "effect->cause"
    Returns:
        A (layers x heads) tensor of average attention proportions
    """
    print(f"\nAnalyzing {direction} attention patterns for delimiter: '{delimiter}'")
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
    
def activation_patching(model, template, dataset):
    """
    Performs activation patching analysis on the model to understand how different
    components contribute to the model's predictions.
    
    This function creates two types of visualizations:
    1. Residual stream activation patching across layers and positions
    2. Attention head output patching across heads and positions
    
    Args:
        model: The transformer model to analyze
        template: String identifier for the current analysis
        dataset: Dictionary containing clean_tokens, corrupted_tokens, and answers
    """
    print(f"\nPerforming activation patching analysis for template: {template}")
    clean_tokens = dataset['clean_tokens']
    corrupted_tokens = dataset['corrupted_tokens']
    answers = dataset['answers']

    answer_tokens = t.concat([
        model.to_tokens(answers, prepend_bos=False).T for answers in answers
    ])

    print(answer_tokens)

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)

    def logit_diff_metric(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens = answer_tokens,
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
    """
    Runs activation patching experiments on various templates of prompts.
    
    Analyzes different types of causal relationships using templates:
    - ALB (Action Location Because)
    - ALS (Action Location So)
    - ALS-2 (Alternative Action Location So)
    - AOS (Action Object So)
    - AOB (Action Object Because)
    
    Args:
        model: The transformer model to analyze
    """
    print("Starting activation patching analysis across multiple templates...")
    template = "ALB"
    dataset = {'clean_tokens': model.to_tokens([
        'John had to dress because he is going to the'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'John had to run because he is going to the', 
                  'John had to rest because he is going to the', 
                  'John had to pack because he is going to the']),
              'answers': [(' dance', ' park'), (' dance', ' gym'), (' dance', ' airport')]
            }

    activation_patching(model, template, dataset)

    template = "ALS"
    dataset = {'clean_tokens': model.to_tokens([
        'Mary went to the store so she wants to'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Mary went to the test so she wants to', 
                  'Mary went to the gym so she wants to', 
                  'Mary went to the library so she wants to']),
              'answers': [(' shop', ' write'), (' shop', ' exercise'), (' shop', ' read')]
            }
    activation_patching(model, template, dataset)

    template = "ALS-2"
    dataset = {'clean_tokens': model.to_tokens([
        'Nadia will be at the beach so she will'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Nadia will be at the library so she will',
                  'Nadia will be at the gym so she will',
                  'Nadia will be at the hospital so she will']),
              'answers': [(' swim', ' read'), (' swim', ' exercise'), (' swim', ' work')]
            }
    activation_patching(model, template, dataset)

    template = "AOS"
    dataset = {'clean_tokens': model.to_tokens([
        'Sara wanted to write so Mark decided to get the'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Sara wanted to go so Mark decided to get the', 
                  'Sara wanted to sleep so Mark decided to get the', 
                  'Sara wanted to play so Mark decided to get the']),
              'answers': [(' book', ' car'), (' book', ' room'), (' book', ' ball')]
            }
    activation_patching(model, template, dataset)

    template = "AOB"
    dataset = {'clean_tokens': model.to_tokens([
        'Jane will read it because John is getting the'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Jane will move it because John is getting the',
                  'Jane will sketch it because John is getting the', 
                  'Jane will play it because John is getting the']),
              'answers': [(' book', ' box'), (' book', ' pencil'), (' book', ' guitar')]
            }
    
    activation_patching(model, template, dataset)

def main():
    """
    Main entry point for the reasoning interpretation analysis.
    
    Sets up the model and directory structure, then runs various analyses
    to understand how the model processes causal relationships.
    """
    print(f"Initializing analysis with model: {model_name}")
    global model_name
    model_name = "gpt2-small"
    if not os.path.exists(f"../results/{model_name}"):
        os.makedirs(f"../results/{model_name}")
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
    # analyze_delimiter_attention(model)
    # analyze_causal_attention(model)
    activation_patching_analysis(model)

if __name__ == "__main__":
    main()
