from datasets import load_dataset


dataset = load_dataset('stereoset', 'intrasentence', split='validation')

print(dataset[0])
