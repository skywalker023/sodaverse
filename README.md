# Welcome to SODAverse 🪐🌟

This is the official repository for our paper:<br>[<b>SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization</b>](https://arxiv.org/abs/2212.10465)<br>

![cosmo-in-soda](assets/cover.jpg)

```
@article{kim2022soda,
    title={SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization},
    author={Hyunwoo Kim and Jack Hessel and Liwei Jiang and Peter West and Ximing Lu and Youngjae Yu and Pei Zhou and Ronan Le Bras and Malihe Alikhani and Gunhee Kim and Maarten Sap and Yejin Choi},
    journal={ArXiv},
    year={2022},
    volume={abs/2212.10465}
}
```

For a brief summary of our paper, please see this [tweet](https://twitter.com/hyunw__kim/status/1605400305126248448).


## 🥤SODA

You can now load SODA from the [HuggingFace hub](https://huggingface.co/datasets/allenai/soda) as the following:
```python
from datasets import load_dataset

dataset = load_dataset("allenai/soda")
```

## 🧑🏻‍🚀COSMO

You can now load COSMO-3B from the [HuggingFace hub](https://huggingface.co/allenai/cosmo-xl) as the following:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/cosmo-xl")
# model.to('cuda')

def set_input(narrative, instruction, dialogue_history):
    input_text = " <turn> ".join(dialogue_history)

    if instruction != "":
        input_text = instruction + " <sep> " + input_text

    if narrative != "":
        input_text = narrative + " <sep> " + input_text
    

    return input_text

def generate(narrative, instruction, dialogue_history):
    """
    narrative: the description of situation/context with the characters included (e.g., "David goes to an amusement park")
    instruction: the perspective/speaker instruction (e.g., "Imagine you are David and speak to his friend Sarah").
    dialogue history: the previous utterances in the dialogue in a list
    """

    input_text = set_input(narrative, instruction, dialogue_history) 

    inputs = tokenizer([input_text], return_tensors="pt")
    # inputs = inputs.to('cuda')
    outputs = model.generate(inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return response

situation = "Cosmo had a really fun time participating in the EMNLP conference at Abu Dhabi."
instruction = "You are Cosmo and you are talking to a friend." # You can also leave the instruction empty

dialogue = [
    "Hey, how was your trip to Abu Dhabi?"
]

response = generate(situation, instruction, dialogue)
print(response)
```

We will also be releasing our 🫧CO<sub>3</sub>! Stay tuned!
