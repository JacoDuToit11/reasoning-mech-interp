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

    if not os.path.exists(f"../results/{model_name}/{category}"):
        os.makedirs(f"../results/{model_name}/{category}")

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

        plt.savefig(f"../results/{model_name}/{category}/activation_patching_per_block_{template_title}.png")

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
    plt.savefig(f"../results/{model_name}/{category}/activation_patching_attn_head_out_all_pos_{template_title}.png")

    # Return the attention head outputs for averaging
    return act_patch_attn_head_out_all_pos

def create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template):
    dataset = {
        'clean_tokens': model.to_tokens([
            base_template.format(token)
            for token, _ in clean_pairs
        ]),
        'corrupted_tokens': model.to_tokens([
            base_template.format(token)
            for token, _ in corrupted_pairs
        ]),
        'answers': [
            (f" {clean[1]}", f" {corrupt[1]}")
            for clean, corrupt in zip(clean_pairs, corrupted_pairs)
        ]
    }

    return dataset

def activation_patching_paper_templates_analysis(model):
    """
    Runs activation patching experiments on various templates of prompts from paper.
    
    Analyzes different types of causal relationships using templates:
    - ALB (Action Location Because)
    - ALS (Action Location So)
    - ALS-2 (Alternative Action Location So)
    - AOS (Action Object So)
    - AOB (Action Object Because)
    
    Args:
        model: The transformer model to analyze
    """
    print("Starting activation patching analysis across semantic templates...")
    category = "paper_templates"
    
    # Store all attention head outputs
    all_attn_head_outputs = []
    
    # Template: "John had to [ACTION] because he is going to the [LOCATION]"
    template_title = "ALB"
    base_template = "John had to {} because he is going to the"
    
    clean_pairs = [
        ("dress", "show"),
        ("shave", "meeting"),
        ("study", "exam")
    ]
    corrupted_pairs = [
        ("work", "office"),
        ("train", "gym"), 
        ("pack", "airport")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Jane will [ACTION] it because John is getting the [OBJECT]"
    template_title = "AOB"
    base_template = "Jane will {} it because John is getting the"
    clean_pairs = [
        ("read", "book"),
        ("eat", "food"),
        ("slice", "bread")
    ]
    corrupted_pairs = [
        ("heat", "pot"),
        ("sketch", "pencil"),
        ("wash", "dish")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Mary went to the [LOCATION] so she wants to [ACTION]"
    template_title = "ALS"
    base_template = "Mary went to the {} so she wants to"
    clean_pairs = [
        ("store", "shop"),
        ("church", "pray"),
        ("airport", "fly")
    ]
    corrupted_pairs = [
        ("exam", "write"),
        ("gym", "exercise"),
        ("library", "read")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Nadia will be at the [LOCATION] so she will [ACTION]"
    template_title = "ALS-2"
    base_template = "Nadia will be at the {} so she will"
    clean_pairs = [
        ("beach", "swim"),
        ("church", "pray"),
        ("airport", "fly")
    ]
    corrupted_pairs = [
        ("library", "read"),
        ("gym", "exercise"),
        ("store", "shop")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Sara wanted to [ACTION] so Mark decided to get the [OBJECT]"
    template_title = "AOS"
    base_template = "Sara wanted to {} so Mark decided to get the"
    clean_pairs = [
        ("study", "book"),
        ("paint", "canvas"),
        ("write", "pen")
    ]
    corrupted_pairs = [
        ("wash", "dish"),
        ("sketch", "pencil"),
        ("cook", "pot"),
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Calculate and save the average
    avg_attn_head_out = t.stack(all_attn_head_outputs).mean(dim=0)
    
    # Plot and save the average
    plt.figure(figsize=(10, 8))
    max_abs_val = abs(avg_attn_head_out.cpu()).max()
    plt.imshow(
        avg_attn_head_out.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,
        vmax=max_abs_val
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Average Attention Head Outputs Across Semantic Templates")
    plt.savefig(f"../results/{model_name}/{category}/activation_patching_attn_head_out_avg.png")
    plt.close()

def activation_patching_mathematical_analysis(model):
    """
    Runs activation patching experiments on mathematical reasoning templates.
    
    Analyzes different types of mathematical relationships using templates:
    - MAB (Mathematical Action Because)
    - MAB-2 (Alternative Mathematical Action Because)
    - MPS (Mathematical Progressive So)
    - MPS-2 (Alternative Mathematical Progressive So)
    - MRS (Mathematical Requirement So)
    
    Args:
        model: The transformer model to analyze
    """
    print("Starting activation patching analysis across mathematical templates...")
    category = "mathematical"
    all_attn_head_outputs = []

    # Template: "John had [X] apples but now has 8 because Mary gave him"
    template_title = "MATH-B-1"
    base_template = "John had {} apples but now has 10 because Mary gave him"
    clean_pairs = [
        ("1", "9"),
        ("8", "2"),
        ("6", "4")
    ]
    corrupted_pairs = [
        ("3", "7"),
        ("5", "5"),
        ("0", "10")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Jane needs [X] apples because she already has 6 and wants a total of"
    template_title = "MATH-B-2"
    base_template = "Jane needs {} apples because she already has 6 and wants a total of"
    clean_pairs = [
        ("4", "10"),
        ("25", "31"),
        ("116", "122")
    ]
    corrupted_pairs = [
        ("3", "9"),
        ("27", "33"),
        ("63", "69")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Mary got [X] oranges so now she has 8 after starting with"
    template_title = "MATH-S-1"
    base_template = "Mary got {} oranges so now she has 8 after starting with"
    clean_pairs = [
        ("3", "5"), 
        ("4", "4"),  
        ("2", "6")  
    ]
    corrupted_pairs = [
        ("5", "3"),  
        ("6", "2"),  
        ("1", "7")  
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Nadia shared [X] bananas so she only has 2 after starting with"
    template_title = "MATH-S-2"
    base_template = "Nadia shared {} bananas so she only has 2 after starting with"
    clean_pairs = [
        ("3", "5"),
        ("4", "6"),
        ("118", "120")
    ]
    corrupted_pairs = [
        ("2", "4"),
        ("19", "21"),
        ("7", "9")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Sarah needed [X] pencils so she could complete her set of 10 after starting with"
    template_title = "MATH-S-3"
    base_template = "Sarah needed {} pencils so she could complete her set of 10 after starting with"
    clean_pairs = [
        ("3", "7"),
        ("5", "5"),
        ("4", "6")
    ]
    corrupted_pairs = [
        ("4", "6"),
        ("2", "8"),
        ("1", "9")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Calculate and save the average
    avg_attn_head_out = t.stack(all_attn_head_outputs).mean(dim=0)
    
    # Plot and save the average
    plt.figure(figsize=(10, 8))
    max_abs_val = abs(avg_attn_head_out.cpu()).max()
    plt.imshow(
        avg_attn_head_out.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,
        vmax=max_abs_val
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Average Attention Head Outputs Across Mathematical Templates")
    plt.savefig(f"../results/{model_name}/{category}/activation_patching_attn_head_out_avg.png")
    plt.close()

def activation_patching_coding_analysis(model):
    """
    Runs activation patching experiments on simple coding reasoning templates.
    
    Analyzes different types of programming logic including:
    - Variable assignment and values
    - Simple conditionals
    - Basic loops
    - Function return values
    - Array/list operations
    
    Both clean and corrupted pairs represent valid code with correct outputs,
    but demonstrate different programming patterns to achieve the results.
    """
    print("Starting activation patching analysis across coding reasoning templates...")
    category = "coding"
    all_attn_head_outputs = []

    # Variable Assignment
    template_title = "VARS"
    base_template = "x = 5\ny = {}\nz = x + y\nz equals"
    clean_pairs = [
        ("3", "8"),    # Adding small numbers
        ("10", "15"),  # Adding larger numbers
        ("0", "5")     # Adding zero
    ]
    corrupted_pairs = [
        ("15", "20"),  # Different but valid additions
        ("25", "30"),
        ("20", "25")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Conditionals
    template_title = "IF"
    base_template = "x = {}\nif x < 10:\n    print('small')\nelse:\n    print('big')\nThe code will print"
    clean_pairs = [
        ("5", "small"),    # Single digit numbers
        ("3", "small"),
        ("2", "small")
    ]
    corrupted_pairs = [
        ("15", "big"),     # Double digit numbers
        ("12", "big"),
        ("11", "big")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Loops
    template_title = "LOOP"
    base_template = "total = 0\nfor i in range({}):\n    total += 2\ntotal equals"
    clean_pairs = [
        ("3", "6"),    # Small ranges
        ("4", "8"),
        ("5", "10")
    ]
    corrupted_pairs = [
        ("10", "20"),  # Larger ranges
        ("8", "16"),
        ("6", "12")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Function Returns
    template_title = "FUNC"
    base_template = "def func(x):\n    return x * {}\n\nresult = func(2)\nresult equals"
    clean_pairs = [
        ("3", "6"),    # Multiplication by small numbers
        ("4", "8"),
        ("5", "10")
    ]
    corrupted_pairs = [
        ("10", "20"),  # Multiplication by larger numbers
        ("8", "16"),
        ("6", "12")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # List Operations
    template_title = "LIST"
    base_template = "nums = [1, 2, {}]\nsum = 0\nfor n in nums:\n    sum += n\nsum equals"
    clean_pairs = [
        ("3", "6"),     # Small numbers in list
        ("4", "7"),
        ("5", "8")
    ]
    corrupted_pairs = [
        ("10", "13"),   # Larger numbers in list
        ("15", "18"),
        ("20", "23")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Calculate and save the average
    avg_attn_head_out = t.stack(all_attn_head_outputs).mean(dim=0)
    
    # Plot and save the average
    plt.figure(figsize=(10, 8))
    max_abs_val = abs(avg_attn_head_out.cpu()).max()
    plt.imshow(
        avg_attn_head_out.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,
        vmax=max_abs_val
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Average Attention Head Outputs Across Coding Templates")
    plt.savefig(f"../results/{model_name}/{category}/activation_patching_attn_head_out_avg.png")
    plt.close()

def activation_patching_coding_logic_analysis(model):
    """
    Runs activation patching experiments on logical coding templates.
    
    Analyzes different types of programming logic including:
    - String operations
    - Boolean logic
    - List operations (non-numeric)
    - Dictionary lookups
    - Control flow
    
    Both clean and corrupted pairs represent valid code with correct outputs,
    but demonstrate different programming patterns to achieve the results.
    """
    print("Starting activation patching analysis across logical coding templates...")
    category = "coding_logic"
    all_attn_head_outputs = []

    # String Operations
    template_title = "STRING"
    base_template = "text = '{}'\nif text.startswith('a'):\n    result = 'yes'\nelse:\n    result = 'no'\nresult equals"
    clean_pairs = [
        ("art", "yes"),      # Simple 'a' words
        ("age", "yes"),
        ("air", "yes")
    ]
    corrupted_pairs = [
        ("dog", "no"),      # Non-'a' words
        ("cat", "no"),
        ("box", "no")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Boolean Logic
    template_title = "BOOL"
    base_template = "is_sunny = {}\nis_warm = True\ncan_swim = is_sunny and is_warm\ncan_swim equals"
    clean_pairs = [
        ("True", "True"),      # Both conditions true
        ("True", "True"),
        ("True", "True")
    ]
    corrupted_pairs = [
        ("False", "False"),    # One condition false
        ("False", "False"),
        ("False", "False")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # List Operations (non-numeric)
    template_title = "LIST"
    base_template = "fruits = ['apple', 'banana', '{}']\nif 'apple' in fruits:\n    result = 'found'\nelse:\n    result = 'missing'\nresult equals"
    clean_pairs = [
        ("orange", "found"),    # Lists with apple
        ("grape", "found"),
        ("mango", "found")
    ]
    corrupted_pairs = [
        ("banana", "found"),    # Different lists with apple
        ("kiwi", "found"),
        ("peach", "found")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Dictionary Operations
    template_title = "DICT"
    base_template = "user = {{'name': '{}', 'active': True}}\nif user['active']:\n    status = 'online'\nelse:\n    status = 'offline'\nstatus equals"
    clean_pairs = [
        ("Alice", "online"),    # Different active users
        ("Bob", "online"),
        ("Charlie", "online")
    ]
    corrupted_pairs = [
        ("David", "online"),    # Other active users
        ("Eve", "online"),
        ("Frank", "online")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Control Flow
    template_title = "FLOW"
    base_template = "status = '{}'\nif status == 'error':\n    msg = 'failed'\nelif status == 'success':\n    msg = 'passed'\nelse:\n    msg = 'unknown'\nmsg equals"
    clean_pairs = [
        ("error", "failed"),     # Error cases
        ("error", "failed"),
        ("error", "failed")
    ]
    corrupted_pairs = [
        ("success", "passed"),   # Success cases
        ("success", "passed"),
        ("success", "passed")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Calculate and save the average
    avg_attn_head_out = t.stack(all_attn_head_outputs).mean(dim=0)
    
    # Plot and save the average
    plt.figure(figsize=(10, 8))
    max_abs_val = abs(avg_attn_head_out.cpu()).max()
    plt.imshow(
        avg_attn_head_out.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,
        vmax=max_abs_val
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Average Attention Head Outputs Across Logical Coding Templates")
    plt.savefig(f"../results/{model_name}/{category}/activation_patching_attn_head_out_avg.png")
    plt.close()

def activation_patching_emotional_analysis(model):
    """
    Runs activation patching experiments on emotional reasoning templates.
    
    Analyzes different types of emotional relationships using templates:
    - EAB (Emotional Action Because)
    - ERS (Emotional Response So)
    - ESB (Emotional State Because)
    - ECS (Emotional Consequence So)
    - ECB (Emotional Cause Because)
    
    Args:
        model: The transformer model to analyze
    """
    print("Starting activation patching analysis across emotional templates...")
    category = "emotional"
    all_attn_head_outputs = []

    # Template: "Tom [ACTION] because Pete made him feel [EMOTION]"
    template_title = "EAB"
    base_template = "Tom {} because Pete made him feel"
    clean_pairs = [
        ("smiled", "happy"),
        ("cried", "sad"),
        ("trembled", "scared")
    ]
    corrupted_pairs = [
        ("frowned", "angry"),
        ("laughed", "amused"),
        ("left", "bad")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Lisa felt [EMOTION] so she decided to [ACTION]"
    template_title = "ERS"
    base_template = "Lisa felt {} so she decided to"
    clean_pairs = [
        ("scared", "hide"),
        ("excited", "dance"),
        ("angry", "shout")
    ]
    corrupted_pairs = [
        ("sad", "cry"),
        ("jealous", "fight"),
        ("happy", "laugh")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Emma feels [EMOTION] because her [OBJECT] is [STATE]"
    template_title = "ESB"
    base_template = "Emma feels {} because her best friend is"
    clean_pairs = [
        ("lonely", "away"),
        ("proud", "successful"),
        ("worried", "sick")
    ]
    corrupted_pairs = [
        ("happy", "here"),
        ("excited", "visiting"),
        ("calm", "sleeping")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Template: "Sarah is [EMOTION] because she got a new [OBJECT]"
    template_title = "ECB"
    base_template = "Sarah is {} because she got a new"
    clean_pairs = [
        ("happy", "puppy"),
        ("nervous", "job"),
        ("excited", "gift")
    ]
    corrupted_pairs = [
        ("sad", "problem"),
        ("proud", "prize"),
        ("worried", "test")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Calculate and save the average
    avg_attn_head_out = t.stack(all_attn_head_outputs).mean(dim=0)
    
    # Plot and save the average
    plt.figure(figsize=(10, 8))
    max_abs_val = abs(avg_attn_head_out.cpu()).max()
    plt.imshow(
        avg_attn_head_out.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,
        vmax=max_abs_val
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Average Attention Head Outputs Across Emotional Templates")
    plt.savefig(f"../results/{model_name}/{category}/activation_patching_attn_head_out_avg.png")
    plt.close()

def activation_patching_physical_analysis(model):
    """
    Runs activation patching experiments on physical causation templates.
    
    Analyzes different types of physical cause-effect relationships using templates:
    - PCB (Physical Cause Because) - Direct physical causes
    - PCS (Physical Consequence So) - Resulting physical states
    - PMB (Physical Material Because) - Material properties
    - PFB (Physical Force Because) - Force and motion
    - PSB (Physical State Because) - Environmental conditions
    
    Args:
        model: The transformer model to analyze
    """
    print("Starting activation patching analysis across physical templates...")

    # The <object> broke because it 
    template = "PCB"
    dataset = {'clean_tokens': model.to_tokens([
        'The glass broke because it fell on the',
        'The metal bent because it was hit with the',
        'The ice melted because it was left in the',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'The glass floated because it fell on the',
                  'The metal sparkled because it was hit with the',
                  'The ice expanded because it was left in the']),
              'answers': [(' concrete', ' carpet'),
                        (' hammer', ' feather'),
                        (' heat', ' shade')]
            }
    activation_patching(model, template, dataset)

    template = "PCS"
    dataset = {'clean_tokens': model.to_tokens([
        'The metal was hot so it turned',
        'The ball was hit so it went',
        'The ice was warm so it turned',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'The metal was new so it turned',
                  'The ball was old so it went',
                  'The ice was cold so it turned']),
              'answers': [(' red', ' grey'), (' up', ' down'), (' soft', ' hard')]
            }
    activation_patching(model, template, dataset)

    template = "PMB"
    dataset = {'clean_tokens': model.to_tokens([
        'The cloth tore because it was too',
        'The rope broke because it was too',
        'The food spoiled because it was too',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'The cloth moved because it was too',
                  'The rope stretched because it was too',
                  'The food changed because it was too']),
              'answers': [(' thin', ' soft'), (' weak', ' long'), (' hot', ' fresh')]
            }
    activation_patching(model, template, dataset)

    template = "PFB"
    dataset = {'clean_tokens': model.to_tokens([
        'The tree fell because the wind was too',
        'The car slid because the road was too',
        'The boat tipped because the sea was too',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'The tree bent because the wind was too',
                  'The car stopped because the road was too',
                  'The boat moved because the sea was too']),
              'answers': [(' strong', ' weak'), (' wet', ' rough'), (' wild', ' calm')]
            }
    activation_patching(model, template, dataset)

    template = "PSB"
    dataset = {'clean_tokens': model.to_tokens([
        'The steel rusted because the air was too',
        'The plant died because the soil was too',
        'The food melted because the room was too',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'The steel shone because the air was too',
                  'The plant grew because the soil was too',
                  'The food froze because the room was too']),
              'answers': [(' wet', ' dry'), (' dry', ' rich'), (' hot', ' cold')]
            }
    activation_patching(model, template, dataset)

def activation_patching_coding_analysis(model):
    """
    Runs activation patching experiments on simple coding reasoning templates.
    
    Analyzes different types of programming logic including:
    - Variable assignment and values
    - Simple conditionals
    - Basic loops
    - Function return values
    - Array/list operations
    
    Both clean and corrupted pairs represent valid code with correct outputs,
    but demonstrate different programming patterns to achieve the results.
    """
    print("Starting activation patching analysis across coding reasoning templates...")
    category = "coding"
    all_attn_head_outputs = []

    # Variable Assignment
    template_title = "VARS"
    base_template = "x = 5\ny = {}\nz = x + y\nz equals"
    clean_pairs = [
        ("3", "8"),    # Adding small numbers
        ("10", "15"),  # Adding larger numbers
        ("0", "5")     # Adding zero
    ]
    corrupted_pairs = [
        ("15", "20"),  # Different but valid additions
        ("25", "30"),
        ("20", "25")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Conditionals
    template_title = "IF"
    base_template = "x = {}\nif x < 10:\n    print('small')\nelse:\n    print('big')\nThe code will print"
    clean_pairs = [
        ("5", "small"),    # Single digit numbers
        ("3", "small"),
        ("2", "small")
    ]
    corrupted_pairs = [
        ("15", "big"),     # Double digit numbers
        ("12", "big"),
        ("11", "big")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Loops
    template_title = "LOOP"
    base_template = "total = 0\nfor i in range({}):\n    total += 2\ntotal equals"
    clean_pairs = [
        ("3", "6"),    # Small ranges
        ("4", "8"),
        ("5", "10")
    ]
    corrupted_pairs = [
        ("10", "20"),  # Larger ranges
        ("8", "16"),
        ("6", "12")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Function Returns
    template_title = "FUNC"
    base_template = "def func(x):\n    return x * {}\n\nresult = func(2)\nresult equals"
    clean_pairs = [
        ("3", "6"),    # Multiplication by small numbers
        ("4", "8"),
        ("5", "10")
    ]
    corrupted_pairs = [
        ("10", "20"),  # Multiplication by larger numbers
        ("8", "16"),
        ("6", "12")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # List Operations
    template_title = "LIST"
    base_template = "nums = [1, 2, {}]\nsum = 0\nfor n in nums:\n    sum += n\nsum equals"
    clean_pairs = [
        ("3", "6"),     # Small numbers in list
        ("4", "7"),
        ("5", "8")
    ]
    corrupted_pairs = [
        ("10", "13"),   # Larger numbers in list
        ("15", "18"),
        ("20", "23")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Calculate and save the average
    avg_attn_head_out = t.stack(all_attn_head_outputs).mean(dim=0)
    
    # Plot and save the average
    plt.figure(figsize=(10, 8))
    max_abs_val = abs(avg_attn_head_out.cpu()).max()
    plt.imshow(
        avg_attn_head_out.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,
        vmax=max_abs_val
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Average Attention Head Outputs Across Coding Templates")
    plt.savefig(f"../results/{model_name}/{category}/activation_patching_attn_head_out_avg.png")
    plt.close()

def activation_patching_transitive_analysis(model):
    """
    Analyzes the model's ability to handle multi-step (transitive) reasoning.
    Tests different types of transitive relationships:
    - Age/Time relationships
    - Spatial/Location relationships
    - Family relationships
    - Comparative relationships
    """
    print("Starting activation patching analysis across transitive reasoning templates...")
    category = "transitive"
    all_attn_head_outputs = []

    # # Age relationships
    # template_title = "AGE"
    # base_template = "Pete is {}. Pete and Andy are the same age. Andy is"
    # clean_pairs = [
    #     ("20", "20"),
    #     ("15", "15"),
    #     ("30", "30")
    # ]
    # corrupted_pairs = [
    #     ("90", "90"),
    #     ("5", "5"),
    #     ("40", "40")
    # ]
    # dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    # attn_head_out = activation_patching(model, category, template_title, dataset)
    # all_attn_head_outputs.append(attn_head_out)

    # Location relationships
    template_title = "LOCATION"
    base_template = "Sara is in the {}. Sara and Tom are in the same place. Tom is in the"
    clean_pairs = [
        ("kitchen", "kitchen"),
        ("library", "library"),
        ("office", "office"),
        ("classroom", "classroom"),
        ("cafeteria", "cafeteria"),
        ("gym", "gym")
    ]
    corrupted_pairs = [
        ("garden", "garden"), 
        ("park", "park"),
        ("bedroom", "bedroom"),
        ("basement", "basement"),
        ("attic", "attic"),
        ("garage", "garage")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    template_title = "COLOR"
    base_template = "The box is {}. The box and ball are the same color. The ball is"
    clean_pairs = [
        ("red", "red"),
        ("blue", "blue"),
        ("green", "green"),
        ("yellow", "yellow"),
        ("purple", "purple"),
        ("orange", "orange")
    ]
    corrupted_pairs = [
        ("white", "white"),
        ("black", "black"),
        ("brown", "brown"),
        ("pink", "pink"),
        ("gray", "gray"),
        ("gold", "gold")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    exit()

    # Family relationships
    # template_title = "FAMILY"
    # base_template = "Jane is Alex's {}. Bob is Jane's son. Alex is Bob's"
    # clean_pairs = [
    #     ("sister", "aunt"),
    #     ("mother", "sister"),
    # ]
    # corrupted_pairs = [
    #     ("aunt", "cousin"),
    #     ("daughter", "grandma")
    # ]
    # dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    # attn_head_out = activation_patching(model, category, template_title, dataset)
    # all_attn_head_outputs.append(attn_head_out)

    # Time relationships
    template_title = "TIME"
    base_template = "Event A happened {} Event B. Event B happened before Event C. Event C happened"
    clean_pairs = [
        ("before", "last"),
        ("after", "second"),
    ]
    corrupted_pairs = [
        ("after", "second"),
        ("before", "last"),
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Cost comparisons
    template_title = "COST"
    base_template = "The book costs {} than the pen. The pen costs more than the pencil. The pencil is the"
    clean_pairs = [
        ("more", "cheapest"),
        ("less", "middle"),
    ]
    corrupted_pairs = [
        ("less", "middle"),
        ("more", "cheapest")
    ]
    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    attn_head_out = activation_patching(model, category, template_title, dataset)
    all_attn_head_outputs.append(attn_head_out)

    # Calculate and save the average
    avg_attn_head_out = t.stack(all_attn_head_outputs).mean(dim=0)
    
    # Plot and save the average
    plt.figure(figsize=(10, 8))
    max_abs_val = abs(avg_attn_head_out.cpu()).max()
    plt.imshow(
        avg_attn_head_out.cpu(),
        cmap="RdBu",
        aspect="auto",
        vmin=-max_abs_val,
        vmax=max_abs_val
    )
    plt.colorbar(label="Value")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Average Attention Head Outputs Across Transitive Templates")
    plt.savefig(f"../results/{model_name}/{category}/activation_patching_attn_head_out_avg.png")
    plt.close()

def main():
    """
    Main entry point for the reasoning interpretation analysis.
    
    Sets up the model and directory structure, then runs various analyses
    to understand how the model processes causal relationships.
    """
    global model_name
    # model_name = "gpt2-small"
    model_name = "gpt2-medium"
    print(f"Initializing analysis with model: {model_name}")
    if not os.path.exists(f"../results/{model_name}"):
        os.makedirs(f"../results/{model_name}")
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name=model_name)
    # analyze_delimiter_attention(model)
    # analyze_causal_attention(model)
    exit()
    global skip_resid_pre 
    skip_resid_pre = True
    activation_patching_paper_templates_analysis(model)
    activation_patching_mathematical_analysis(model)
    # activation_patching_emotional_analysis(model)
    # activation_patching_physical_analysis(model)
    # activation_patching_arithmetic_analysis(model)
    # activation_patching_transitive_analysis(model)

    # activation_patching_coding_logic_analysis(model)

# NOTES:
# Max has 4 apples, tom has 6, who has more? Combine math and semantic reasoning
# Coding logic!
# write code for math type reasoning tasks

if __name__ == "__main__":
    main()
