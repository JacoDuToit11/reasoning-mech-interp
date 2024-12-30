import openai
import os
from os import environ

openai.api_key = environ.get("OPENAI_API_KEY")
data_dir = "../data"

type_to_templates = {
    "B->A": [
        "Alice went to <location> because she wanted to <verb> <object>",
        "Alice play <object> because she enjoys <verb> <object>",
        "Alice went to <location> because <location> is a good place for <object>",
    ],
    "A->B": [
        "Bob and Chris got work to do so they are <adjective> to <verb>",
        "Bob and Chris made <object> so <pronoun> are <adjective1> and <adjective2>",
    ],
    "non-causal": [
        "Alice went to <location> and she <verb> <object>",
        "Alice play <object> and <pronoun> is <adjective>",
        "Bob and Chris got work to do but they are <adjective> to <verb>",
        "Bob and Chris made <object> while <pronoun> are <adjective1> and <adjective2>",
    ],
    "random": [
        "Alice went to <random> because she <verb> <object>",
        "Alice play <object> because she enjoys <verb> <random>",
        "Alice went to <location> and <location> is <adjective>",
        "Bob and Chris made <object> so <pronoun> are <random> and <adjective2>",
        "Bob and Chris got work to do so they are <random> to <verb>",
    ],
}

for template_type, templates in type_to_templates.items():
    if os.path.exists(f"{data_dir}/{template_type}_filled.txt"):
        os.remove(f"{data_dir}/{template_type}_filled.txt")
        os.remove(f"{data_dir}/{template_type}_filled_list.txt")

    for template in templates:
        prompt = (
            f"Fill in the placeholders in this template:\n{template}\n\n"
            "Replace <location> with a random place, <object> with a random noun, "
            "<verb> with a random verb, <pronoun> with a random pronoun, "
            "<adjective> with a random adjective, and <random> with any random word.\n"
            "Only return the completed sentence with no extra text."
        )

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt}],
        )

        responses = {}
        responses[template] = response.choices[0].message.content
        with open(f"{data_dir}/{template_type}_filled.txt", "a") as f:
            f.write(f"{responses[template]}\n")

    with open(f"{data_dir}/{template_type}_filled.txt", "r") as f:
        all_responses = [line.strip() for line in f.readlines()]
    
    with open(f"{data_dir}/{template_type}_filled_list.txt", "w") as f:
        f.write(str(all_responses))
