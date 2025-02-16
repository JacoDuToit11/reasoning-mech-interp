import os
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from transformer_lens import patching
from jaxtyping import Float
from torch import Tensor
from pathlib import Path
import json

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
    plt.savefig(f"{output_dir}/{model_name}/attention_map_delim_{delimiter.strip()}.png")

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
    plt.savefig(f"{output_dir}/{model_name}/attention_map_cause_effect_{delimiter.strip()}.png")

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
    
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    per_prompt: bool = False,
) -> Float[Tensor, "*batch"]:
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def plot_and_save_attention_heatmap(attention_data, title, category, model_name, filename):
    """
    Plots and saves an attention head heatmap visualization.
    
    Args:
        attention_data: Tensor containing attention head data
        title: Title for the plot
        category: Category folder for saving results
        model_name: Name of the model being analyzed
        filename: Name for the saved file
    """
    plt.figure(figsize=(10, 8))
    max_abs_val = abs(attention_data.cpu()).max()
    plt.imshow(
        attention_data.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,
        vmax=max_abs_val
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(title)
    plt.savefig(f"{output_dir}/{model_name}/{category}/{filename}.png")
    plt.close()

def activation_patching(model, category, template_title, dataset):
    """
    Performs activation patching analysis on the model to understand how different
    components contribute to the model's predictions.
    
    This function creates two types of visualizations:
    1. Residual stream activation patching across layers and positions
    2. Attention head output patching across heads and positions
    
    Args:
        model: The transformer model to analyze
        category: String identifier for the current analysis
        template_title: String identifier for the current template
        dataset: Dictionary containing clean_tokens, corrupted_tokens, and answers
    """

    if not os.path.exists(f"{output_dir}/{model_name}/{category}"):
        os.makedirs(f"{output_dir}/{model_name}/{category}")

    print(f"\nPerforming activation patching analysis for template: {template_title}")
    clean_tokens = dataset['clean_tokens']
    corrupted_tokens = dataset['corrupted_tokens']
    answers = dataset['answers']

    answer_tokens = t.concat([
        model.to_tokens(answers, prepend_bos=False).T for answers in answers
    ])

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
    print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
    
    def logit_diff_metric(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
        corrupted_logit_diff: float = corrupted_logit_diff,
        clean_logit_diff: float = clean_logit_diff,
    ) -> Float[Tensor, ""]:
        """
        Linear function of logit diff, calibrated so that it equals 0 when performance is same as on corrupted input, and 1
        when performance is same as on clean input.
        """
        patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
        return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

    if not skip_resid_pre:
    
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

        plt.savefig(f"{output_dir}/{model_name}/{category}/activation_patching_per_block_{template_title}.png")

    act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
        model,
        corrupted_tokens,
        clean_cache,
        logit_diff_metric
    )

    # Get the maximum absolute value from the data
    max_abs_val = abs(act_patch_attn_head_out_all_pos.cpu()).max()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(
        act_patch_attn_head_out_all_pos.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,  # Symmetric negative bound
        vmax=max_abs_val    # Symmetric positive bound
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Attention Head Outputs Heatmap")
    plt.savefig(f"{output_dir}/{model_name}/{category}/activation_patching_attn_head_out_all_pos_{template_title}.png")

    # Return the attention head outputs for averaging
    return act_patch_attn_head_out_all_pos

def create_patching_dataset(model, pairs, base_template):
    """
    Creates a dataset where each pair is tested against all other pairs as corrupted versions.
    All pairs are processed together in a batch.
    
    Args:
        model: The transformer model
        pairs: List of [token, completion] pairs
        base_template: The template string to format
        
    Returns:
        Dictionary containing:
            - clean_tokens: Batch of tokens for all clean versions
            - corrupted_tokens: Batch of tokens for all corrupted versions
            - answers: List of (correct, incorrect) answer pairs
    """
    n_pairs = len(pairs)
    
    # Create all possible combinations of clean and corrupted pairs
    clean_prompts = []
    corrupted_prompts = []
    answer_pairs = []
    
    for i in range(n_pairs):
        for j in range(n_pairs):
            if i != j:  # Skip pairing an example with itself
                clean_prompts.append(base_template.format(pairs[i][0]))
                corrupted_prompts.append(base_template.format(pairs[j][0]))
                answer_pairs.append((f" {pairs[i][1]}", f" {pairs[j][1]}"))
    
    # Convert prompts to tokens
    clean_tokens = model.to_tokens(clean_prompts)
    corrupted_tokens = model.to_tokens(corrupted_prompts)
    
    return {
        'clean_tokens': clean_tokens,
        'corrupted_tokens': corrupted_tokens,
        'answers': answer_pairs
    }

def load_templates():
    """Load templates from JSON file in data directory."""
    project_root = Path(__file__).parent.parent
    template_path = project_root / 'data' / 'templates.json'
    
    with open(template_path, 'r') as f:
        return json.load(f)

def activation_patching_template_analysis(model, category: str):
    """
    Runs activation patching experiments on templates from a specified category.
    """
    print(f"Starting activation patching analysis across {category}...")
    templates = load_templates()
    
    # Store all attention head outputs
    all_attn_head_outputs = []
    
    # Process each template
    for template_config in templates[category].values():
        template_title = template_config["title"]
        base_template = template_config["base_template"]
        pairs = template_config["pairs"]
        
        # Get datasets for each clean pair tested against all other pairs
        dataset = create_patching_dataset(model, pairs, base_template)
        
        # Run activation patching for this dataset
        attn_head_out = activation_patching(
            model, 
            category, 
            template_title, 
            dataset
        )
        
        # Append results for this template
        all_attn_head_outputs.append(attn_head_out)

    # Calculate and save the average across all templates
    avg_attn_head_out = t.stack(all_attn_head_outputs).mean(dim=0)
    
    plot_and_save_attention_heatmap(
        attention_data=avg_attn_head_out,
        title=f"Average Attention Head Outputs Across {category.replace('_', ' ').title()}",
        category=category,
        model_name=model_name,
        filename="activation_patching_attn_head_out_avg"
    )

def main():
    """
    Main entry point for the reasoning interpretation analysis.
    
    Sets up the model and directory structure, then runs various analyses
    to understand how the model processes causal relationships.
    """
    global model_name, output_dir, skip_resid_pre
    model_name = "gpt2-small"
    # model_name = "gpt2-medium"
    output_dir = "../results"
    skip_resid_pre = True

    global device
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Initializing analysis with model: {model_name}")
    if not os.path.exists(f"{output_dir}/{model_name}"):
        os.makedirs(f"{output_dir}/{model_name}")
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name=model_name)
    # analyze_delimiter_attention(model)
    # analyze_causal_attention(model)
    activation_patching_template_analysis(model, "paper_templates")
    # activation_patching_template_analysis(model, "mathematical_templates")

if __name__ == "__main__":
    main()
