import OpenAttack as oa
import datasets # use the Hugging Face's datasets library

### map the continous value to categorical 
# change the SST dataset into 2-class
def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }


"""
Step 1. Customize the victim model
Could be implemented by adding the openattack classifier
"""

# choose a trained victim classification model
victim = oa.DataManager.loadVictim("BERT.SST")

"""
Step 2. Customize dataset
Like this:
dataset = datasets.Dataset.from_dict({
        "x": [
            "I hate this movie.",
            "I like this apple."
        ],
        "y": [
            0, # 0 for negative
            1, # 1 for positive
        ]
    })
"""
# choose 20 examples from SST-2 as the evaluation data 
dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)


"""
Step 3. The attacker configuration
tokenizer, (ESM alphabet)
substitute, (BLOSUM 62)
token_unk,
filter_words,
lang, # must create the protein language
"""
# choose PWWS as the attacker and initialize it with default parameters
attacker = oa.attackers.PWWSAttacker(
    
)



"""
Step 4. Prepare the attacker. 
attacker, ## step 3
victim, ## done
lang, # protein language
tokenizer, # alphabet
invoke_limit, # could be set
metrics, # try the original ones
"""
# prepare for attacking
attack_eval = oa.AttackEval(attacker, victim)
# launch attacks and print attack results 
attack_eval.eval(dataset, visualize=True)