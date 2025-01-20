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

def create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template):
    dataset = {
        'clean_tokens': model.to_tokens([
            base_template.format(action)
            for action, _ in clean_pairs
        ]),
        'corrupted_tokens': model.to_tokens([
            base_template.format(action)
            for action, _ in corrupted_pairs
        ]),
        'answers': [
            (f" {clean[1]}", f" {corrupt[1]}")
            for clean, corrupt in zip(clean_pairs, corrupted_pairs)
        ]
    }

    return dataset

def activation_patching_semantic_analysis(model):
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
    print("Starting activation patching analysis across semantic templates...")
    # Template: "John had to {action} because he is going to the {location}"
    template_title = "ALB"
    base_template = "John had to {} because he is going to the"
    
    clean_pairs = [
        ("dress", "dance"),
        ("pray", "church"),
        ("study", "test")
    ]
    corrupted_pairs = [
        ("run", "park"),
        ("rest", "gym"), 
        ("pack", "airport")
    ]

    dataset = create_patching_dataset(model, clean_pairs, corrupted_pairs, base_template)
    activation_patching(model, template_title, dataset)

    exit()

    template = "ALS"
    dataset = {'clean_tokens': model.to_tokens([
        'Mary went to the store so she wants to',
        'Mary went to the church so she wants to',
        'Mary went to the airport so she wants to',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Mary went to the test so she wants to', 
                  'Mary went to the gym so she wants to', 
                  'Mary went to the library so she wants to']),
              'answers': [(' shop', ' write'), (' pray', ' exercise'), (' fly', ' read')]
            }
    activation_patching(model, template, dataset)

    template = "ALS-2"
    dataset = {'clean_tokens': model.to_tokens([
        'Nadia will be at the beach so she will',
        'Nadia will be at the church so she will',
        'Nadia will be at the airport so she will',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Nadia will be at the library so she will',
                  'Nadia will be at the gym so she will',
                  'Nadia will be at the hospital so she will']),
              'answers': [(' swim', ' read'), (' pray', ' exercise'), (' fly', ' work')]
            }
    activation_patching(model, template, dataset)

    template = "AOS"
    dataset = {'clean_tokens': model.to_tokens([
        'Sara wanted to write so Mark decided to get the',
        'Sara wanted to pray so Mark decided to get the',
        'Sara wanted to study so Mark decided to get the',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Sara wanted to go so Mark decided to get the', 
                  'Sara wanted to sleep so Mark decided to get the', 
                  'Sara wanted to play so Mark decided to get the']),
              'answers': [(' book', ' car'), (' bible', ' room'), (' book', ' guitar')]
            }
    activation_patching(model, template, dataset)

    template = "AOB"
    dataset = {'clean_tokens': model.to_tokens([
        'Jane will read it because John is getting the',
        'Jane will eat it because John is getting the',
        'Jane will throw it because John is getting the',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Jane will move it because John is getting the',
                  'Jane will sketch it because John is getting the', 
                  'Jane will play it because John is getting the']),
              'answers': [(' book', ' box'), (' food', ' pencil'), (' ball', ' guitar')]
            }
    
    activation_patching(model, template, dataset)

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

    template = "MAB"
    dataset = {'clean_tokens': model.to_tokens([
        'John had 5 apples but now has 8 because Mary gave him',
        'John had 3 apples but now has 7 because Mary gave him',
        'John had 12 apples but now has 19 because Mary gave him',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'John had 3 apples but now has 8 because Mary gave him',
                  'John had 6 apples but now has 7 because Mary gave him',
                  'John had 4 apples but now has 19 because Mary gave him']),
              'answers': [(' 3', ' 5'), (' 4', ' 1'), (' 7', ' 15')]
            }
    activation_patching(model, template, dataset)

    template = "MAB-2"
    dataset = {'clean_tokens': model.to_tokens([
        'Jane needs 4 apples because she already has 6 and wants a total of',
        'Jane needs 10 apples because she already has 21 and wants a total of',
        'Jane needs 110 apples because she already has 12 and wants a total of'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Jane needs 3 apples because she already has 6 and wants a total of',
                  'Jane needs 12 apples because she already has 21 and wants a total of',
                  'Jane needs 57 apples because she already has 12 and wants a total of']),
              'answers': [(' 10', ' 9'), (' 31', ' 33'), (' 122', ' 69')]
            }
    activation_patching(model, template, dataset)

    template = "MPS"
    dataset = {'clean_tokens': model.to_tokens([
        'Mary got 3 oranges so now she has 8 after starting with',
        'Mary got 12 oranges so now she has 19 after starting with',
        'Mary got 3 oranges so now she has 21 after starting with'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Mary got 2 oranges so now she has 8 after starting with',
                  'Mary got 6 oranges so now she has 19 after starting with',
                  'Mary got 4 oranges so now she has 21 after starting with']),
              'answers': [(' 5', ' 6'), (' 7', ' 13'), (' 18', ' 17')]
            }
    activation_patching(model, template, dataset)

    template = "MPS-2"
    dataset = {'clean_tokens': model.to_tokens([
        'Nadia shared 3 bananas so she only has 2 after starting with',
        'Nadia shared 4 bananas so she only has 3 after starting with',
        'Nadia shared 5 bananas so she only has 4 after starting with'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Nadia shared 2 bananas so she only has 2 after starting with',
                  'Nadia shared 6 bananas so she only has 3 after starting with',
                  'Nadia shared 3 bananas so she only has 4 after starting with']),
              'answers': [(' 5', ' 4'), (' 7', ' 9'), (' 9', ' 7')]
            }
    activation_patching(model, template, dataset)

    template = "MRS"
    dataset = {'clean_tokens': model.to_tokens([
        'Sarah needed 3 pencils so she could complete her set of 10 after starting with',
        'Sarah needed 5 pencils so she could complete her set of 12 after starting with',
        'Sarah needed 4 pencils so she could complete her set of 15 after starting with'
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Sarah needed 4 pencils so she could complete her set of 10 after starting with',
                  'Sarah needed 3 pencils so she could complete her set of 12 after starting with',
                  'Sarah needed 6 pencils so she could complete her set of 15 after starting with']),
              'answers': [(' 7', ' 6'), (' 7', ' 9'), (' 11', ' 9')]
            }
    activation_patching(model, template, dataset)

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

    # Template: "Tom {emotion} because Pete {action}"
    template = "EAB"
    base_template = "Tom {} because Pete was {}"

    clean_emotions = [
        ("laughed", "joked"),
        ("cried", "yelled"), 
        ("frowned", "lied")
    ]
    corrupted_emotions = [
        ("cried", "yelled"),
        ("smiled", "laughed"), 
        ("frowned", "cried")
    ]

    answers = [
        (' joked', ' slept'),
        (' yelled', ' smiled'),
        (' lied', ' waved')
    ]

    print(clean_emotions)
    exit()
    
    dataset = {
        'clean_tokens': model.to_tokens([
            base_template.format(emotion) 
            for emotion in clean_emotions
        ]),
        'corrupted_tokens': model.to_tokens([
            base_template.format(emotion)
            for emotion in corrupted_emotions
        ]),
        'answers': answers
    }

    print(dataset)
    activation_patching(model, template, dataset)

    exit()

    template = "ERS"
    dataset = {'clean_tokens': model.to_tokens([
        'Lisa felt scared so she decided to',
        'Lisa felt excited so she decided to',
        'Lisa felt angry so she decided to',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Lisa felt tired so she decided to',
                  'Lisa felt hungry so she decided to',
                  'Lisa felt cold so she decided to']),
              'answers': [(' hide', ' sleep'), (' dance', ' eat'), (' shout', ' warm')]
            }
    activation_patching(model, template, dataset)

    template = "ESB"
    dataset = {'clean_tokens': model.to_tokens([
        'Emma feels lonely because her best friend is',
        'Emma feels proud because her project is',
        'Emma feels worried because her test is',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Emma feels cold because her best friend is',
                  'Emma feels tired because her project is',
                  'Emma feels hungry because her test is']),
              'answers': [(' away', ' here'), (' perfect', ' late'), (' tomorrow', ' today')]
            }
    activation_patching(model, template, dataset)

    template = "ECS"
    dataset = {'clean_tokens': model.to_tokens([
        'David received good news so he felt very',
        'David lost his wallet so he felt very',
        'David won the race so he felt very',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'David read a book so he felt very',
                  'David ate lunch so he felt very',
                  'David took a walk so he felt very']),
              'answers': [(' happy', ' tired'), (' upset', ' full'), (' proud', ' relaxed')]
            }
    activation_patching(model, template, dataset)

    template = "ECB"
    dataset = {'clean_tokens': model.to_tokens([
        'Sarah is happy because she got a new',
        'Sarah is nervous because she has a big',
        'Sarah is excited because she won the',
        ]),
              'corrupted_tokens': model.to_tokens([
                  'Sarah is tired because she got a new',
                  'Sarah is hungry because she has a big',
                  'Sarah is cold because she won the']),
              'answers': [(' puppy', ' book'), (' test', ' lunch'), (' prize', ' coat')]
            }
    activation_patching(model, template, dataset)

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

def main():
    """
    Main entry point for the reasoning interpretation analysis.
    
    Sets up the model and directory structure, then runs various analyses
    to understand how the model processes causal relationships.
    """
    global model_name
    model_name = "gpt2-small"
    # model_name = "gpt2-medium"
    print(f"Initializing analysis with model: {model_name}")
    if not os.path.exists(f"../results/{model_name}"):
        os.makedirs(f"../results/{model_name}")
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
    # analyze_delimiter_attention(model)
    # analyze_causal_attention(model)
    activation_patching_semantic_analysis(model)
    # activation_patching_mathematical_analysis(model)
    # activation_patching_emotional_analysis(model)
    # activation_patching_physical_analysis(model)


if __name__ == "__main__":
    main()
