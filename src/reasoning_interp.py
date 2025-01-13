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
    
def activation_patching(model, template, dataset):
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

    # template = "ALB"
    # dataset = {'clean_tokens': model.to_tokens(['John had to dress because he is going to the']),
    #           'corrupted_tokens': model.to_tokens(['John had to run because he is going to the', 'John had to rest because he is going to the', 'John had to pack because he is going to the']),
    #           'answers': [(' dance', ' park'), (' dance', ' gym'), (' dance', ' airport')]
    #         }

    # activation_patching(model, template, dataset)

    # template = "ALS"
    # dataset = {'clean_tokens': model.to_tokens(['Mary went to the store so she wants to']),
    #           'corrupted_tokens': model.to_tokens(['Mary went to the test so she wants to', 'Mary went to the gym so she wants to', 'Mary went to the library so she wants to']),
    #           'answers': [(' shop', ' write'), (' shop', ' exercise'), (' shop', ' read')]
    #         }
    # activation_patching(model, template, dataset)

    template = "AOS"
    dataset = {'clean_tokens': model.to_tokens([
        'Sara wanted to write so Mark decided to get the']),
              'corrupted_tokens': model.to_tokens([
                  'Sara wanted to go so Mark decided to get the', 
                  'Sara wanted to sleep so Mark decided to get the', 
                  'Sara wanted to play so Mark decided to get the']),
              'answers': [(' book', ' car'), (' book', ' room'), (' book', ' ball')]
            }
    activation_patching(model, template, dataset)

    template = "AOB"
    dataset = {'clean_tokens': model.to_tokens([
        "Jane will read it because John is getting the"
        f]),
              'corrupted_tokens': model.to_tokens([
        "Jane will move it because John is getting the",
        "Jane will sketch it because John is getting the", 
        "Jane will break it because John is getting the"]),
              'answers': [(' book', ' box'), (' paper', ' pencil'), (' tools', ' parts')]
            }
    activation_patching(model, template, dataset)

    # template = "AOB"
    # clean_tokens = model.to_tokens(["Jane will eat it because John is getting the", "Jane will eat it because John is getting the"])
    # corrupted_tokens = model.to_tokens(["Jane will sing it because John is getting the", "Jane will sing it because John is getting the"])

    # clean_answers = [(" sandwich", " music")]
    # corr_answers = [(" sandwich", " music")]
    # activation_patching(model, template, clean_tokens, corrupted_tokens, clean_answers, corr_answers)

    template = "AOB"
    clean_tokens = model.to_tokens([
        "Jane will read it because John is getting the.",
        "Jane will write it because John is getting the.",
        "Jane will fix it because John is getting the.",
        # "Jane will return it because John is getting the.",
        # "Jane will paint it because John is getting the.",
        # "Jane will pack it because John is getting the.",
        # "Jane will clean it because John is getting the.",
        # "Jane will assemble it because John is getting the.",
        # "Jane will buy it because John is getting the.",
        # "Jane will sell it because John is getting the.",
        # "Jane will cook it because John is getting the.",
        # "Jane will open it because John is getting the.",
        # "Jane will wrap it because John is getting the.",
        # "Jane will test it because John is getting the.",
        # "Jane will charge it because John is getting the.",
        # "Jane will debug it because John is getting the.",
        # "Jane will lock it because John is getting the.",
        # "Jane will unlock it because John is getting the.",
        # "Jane will decorate it because John is getting the.",
        # "Jane will scan it because John is getting the."
    ])

    corrupted_tokens = model.to_tokens([
        "Jane will move it because John is getting the.",
        "Jane will sketch it because John is getting the.",
        "Jane will break it because John is getting the.",
        # "Jane will borrow it because John is getting the.",
        # "Jane will polish it because John is getting the.",
        # "Jane will unpack it because John is getting the.",
        # "Jane will dirty it because John is getting the.",
        # "Jane will disassemble it because John is getting the.",
        # "Jane will steal it because John is getting the.",
        # "Jane will trade it because John is getting the.",
        # "Jane will burn it because John is getting the.",
        # "Jane will close it because John is getting the.",
        # "Jane will unwrap it because John is getting the.",
        # "Jane will ignore it because John is getting the.",
        # "Jane will discharge it because John is getting the.",
        # "Jane will confuse it because John is getting the.",
        # "Jane will leave it because John is getting the.",
        # "Jane will seal it because John is getting the.",
        # "Jane will remove it because John is getting the.",
        # "Jane will copy it because John is getting the."
    ])

    clean_answers = [
        (" book", " tool"),
        (" pen", " paper"),
        (" device", " part"),
        # (" item", " gift"),
        # (" canvas", " brush"),
        # (" suitcase", " backpack"),
        # (" table", " cloth"),
        # (" furniture", " screws"),
        # (" groceries", " cart"),
        # (" car", " money"),
        # (" meal", " ingredients"),
        # (" envelope", " letter"),
        # (" box", " ribbon"),
        # (" code", " laptop"),
        # (" phone", " charger"),
        # (" program", " computer"),
        # (" door", " key"),
        # (" safe", " lock"),
        # (" house", " decorations"),
        # (" document", " printer")
    ]

    corr_answers = [
        (" tool", " book"),
        (" paper", " pen"),
        (" part", " device"),
        # (" gift", " item"),
        # (" brush", " canvas"),
        # (" backpack", " suitcase"),
        # (" cloth", " table"),
        # (" screws", " furniture"),
        # (" cart", " groceries"),
        # (" money", " car"),
        # (" ingredients", " meal"),
        # (" letter", " envelope"),
        # (" ribbon", " box"),
        # (" laptop", " code"),
        # (" charger", " phone"),
        # (" computer", " program"),
        # (" key", " door"),
        # (" lock", " safe"),
        # (" decorations", " house"),
        # (" printer", " document")
        ]
    
    activation_patching(model, template, clean_tokens, corrupted_tokens, clean_answers, corr_answers)


    template = "ALS"
    clean_tokens = model.to_tokens([
        "Mary went to the store so she wants to shop.",
        "Mary went to the gym so she wants to exercise.",
        "Mary went to the library so she wants to study.",
        # "Mary went to the park so she wants to relax.",
        # "Mary went to the office so she wants to work.",
        # "Mary went to the beach so she wants to swim.",
        # "Mary went to the kitchen so she wants to cook.",
        # "Mary went to the garden so she wants to plant.",
        # "Mary went to the mall so she wants to browse.",
        # "Mary went to the airport so she wants to travel.",
        # "Mary went to the cinema so she wants to watch."
    ])

    corrupted_tokens = model.to_tokens([
        "Mary went to the store so she wants to run.",
        "Mary went to the gym so she wants to rest.",
        "Mary went to the library so she wants to chat.",
        # "Mary went to the park so she wants to work.",
        # "Mary went to the office so she wants to cook.",
        # "Mary went to the beach so she wants to read.",
        # "Mary went to the kitchen so she wants to clean.",
        # "Mary went to the garden so she wants to play.",
        # "Mary went to the mall so she wants to exercise.",
        # "Mary went to the airport so she wants to relax.",
        # "Mary went to the cinema so she wants to eat."
    ])

    clean_answers = [
        (" store", " shop"),
        (" gym", " exercise"),
        (" library", " study"),
        # (" park", " relax"),
        # (" office", " work"),
        # (" beach", " swim"),
        # (" kitchen", " cook"),
        # (" garden", " plant"),
        # (" mall", " browse"),
        # (" airport", " travel"),
        # (" cinema", " watch")
    ]

    corr_answers = [
        (" store", " run"),
        (" gym", " rest"),
        (" library", " chat"),
        # (" park", " work"),
        # (" office", " cook"),
        # (" beach", " read"),
        # (" kitchen", " clean"),
        # (" garden", " play"),
        # (" mall", " exercise"),
        # (" airport", " relax"),
        # (" cinema", " eat")
    ]

    activation_patching(model, template, clean_tokens, corrupted_tokens, clean_answers, corr_answers)

    template = "ALS-2"
    clean_tokens = model.to_tokens([
        "Nadia will be at the park so she will jog.",
        "Nadia will be at the kitchen so she will cook.",
        "Nadia will be at the office so she will work.",
        # "Nadia will be at the library so she will read.",
        # "Nadia will be at the gym so she will exercise.",
        # "Nadia will be at the beach so she will swim.",
        # "Nadia will be at the store so she will shop.",
        # "Nadia will be at the airport so she will travel.",
        # "Nadia will be at the garden so she will plant.",
        # "Nadia will be at the theater so she will perform.",
        # "Nadia will be at the stadium so she will cheer."
    ])

    corrupted_tokens = model.to_tokens([
        "Nadia will be at the park so she will nap.",
        "Nadia will be at the kitchen so she will clean.",
        "Nadia will be at the office so she will relax.",
        # "Nadia will be at the library so she will nap.",
        # "Nadia will be at the gym so she will stretch.",
        # "Nadia will be at the beach so she will sunbathe.",
        # "Nadia will be at the store so she will browse.",
        # "Nadia will be at the airport so she will wait.",
        # "Nadia will be at the garden so she will rest.",
        # "Nadia will be at the theater so she will watch.",
        # "Nadia will be at the stadium so she will play."
    ])

    clean_answers = [
        (" park", " jog"),
        (" kitchen", " cook"),
        (" office", " work"),
        # (" library", " read"),
        # (" gym", " exercise"),
        # (" beach", " swim"),
        # (" store", " shop"),
        # (" airport", " travel"),
        # (" garden", " plant"),
        # (" theater", " perform"),
        # (" stadium", " cheer")
    ]

    corr_answers = [
        (" park", " nap"),
        (" kitchen", " clean"),
        (" office", " relax"),
        # (" library", " nap"),
        # (" gym", " stretch"),
        # (" beach", " sunbathe"),
        # (" store", " browse"),
        # (" airport", " wait"),
        # (" garden", " rest"),
        # (" theater", " watch"),
        # (" stadium", " play")
    ]

    activation_patching(model, template, clean_tokens, corrupted_tokens, clean_answers, corr_answers)

    template = "AOS"
    clean_tokens = model.to_tokens([
        "Sarah wanted to read so Mark decided to get the book.",
        "Sarah wanted to write so Mark decided to get the pen.",
        "Sarah wanted to paint so Mark decided to get the brush.",
        # "Sarah wanted to cook so Mark decided to get the pan.",
        # "Sarah wanted to fix so Mark decided to get the tool.",
        # "Sarah wanted to swim so Mark decided to get the swimsuit.",
        # "Sarah wanted to clean so Mark decided to get the mop.",
        # "Sarah wanted to plant so Mark decided to get the seeds.",
        # "Sarah wanted to travel so Mark decided to get the ticket.",
        # "Sarah wanted to study so Mark decided to get the textbook."
    ])

    corrupted_tokens = model.to_tokens([
        "Sarah wanted to read so Mark decided to get the lamp.",
        "Sarah wanted to write so Mark decided to get the paper.",
        "Sarah wanted to paint so Mark decided to get the canvas.",
        # "Sarah wanted to cook so Mark decided to get the knife.",
        # "Sarah wanted to fix so Mark decided to get the nail.",
        # "Sarah wanted to swim so Mark decided to get the goggles.",
        # "Sarah wanted to clean so Mark decided to get the cloth.",
        # "Sarah wanted to plant so Mark decided to get the shovel.",
        # "Sarah wanted to travel so Mark decided to get the bag.",
        # "Sarah wanted to study so Mark decided to get the lamp."
    ])

    clean_answers = [
        (" read", " book"),
        (" write", " pen"),
        (" paint", " brush"),
        # (" cook", " pan"),
        # (" fix", " tool"),
        # (" swim", " swimsuit"),
        # (" clean", " mop"),
        # (" plant", " seeds"),
        # (" travel", " ticket"),
        # (" study", " textbook")
    ]

    corr_answers = [
        (" read", " lamp"),
        (" write", " paper"),
        (" paint", " canvas"),
        # (" cook", " knife"),
        # (" fix", " nail"),
        # (" swim", " goggles"),
        # (" clean", " cloth"),
        # (" plant", " shovel"),
        # (" travel", " bag"),
        # (" study", " lamp")
    ]

    activation_patching(model, template, clean_tokens, corrupted_tokens, clean_answers, corr_answers)

def main():
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
