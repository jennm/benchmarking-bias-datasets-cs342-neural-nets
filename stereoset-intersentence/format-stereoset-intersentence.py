from datasets import load_dataset
import pandas as pd


def convert_to_prompt(context, sentence_options):
    # prompt = f"""Choose the appropriate option given the context:
    prompt = f"Context: {context}"
    for i in range(len(sentence_options)):
        prompt += f"\nOption {i}: {sentence_options[i]}"
    return prompt


dataset = load_dataset("stereoset", 'intersentence', split='validation')


dataframe = pd.DataFrame(
    columns=['context', 'bias_type', 'target', 'gold_label'])

print(len(dataset))

data_list = list()
for i in range(len(dataset)):
    prompt = convert_to_prompt(
        dataset[i]['context'], dataset[i]['sentences']['sentence'])
    data_list.append(pd.Series(
        [prompt, dataset[i]['bias_type'], dataset[i]['target'], dataset[i]['sentences']['gold_label']], index=dataframe.columns))

dataframe = pd.concat(data_list, axis=1).T

dataframe.to_csv('intersentence-formatted-data.csv', index=False)
