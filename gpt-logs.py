from transformers import AutoTokenizer, RobertaForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits


sentence = ["HuggingFace", "is", "a", "company",
            "based", "in", "Paris", "and", "New", "York"]
mask_index = len(sentence) - 1
str_sentence = ""

for i in range(len(sentence)):
    if i != mask_index:
        word = sentence[i]
    else:
        word = "<mask>"
    str_sentence += f" {word}"

print(str_sentence)
inputs = tokenizer(
    str_sentence[1:], return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

print("bye")
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[
    0].nonzero(as_tuple=True)[0]
print("hi", mask_token_index)
print(logits)
hs = logits[0][mask_index]
print(inputs)
print("mask id", mask_index)
# mask_id = tokenizer.convert_tokens_to_ids(inputs.input_ids[])
mask_id = tokenizer.mask_token_id
print(inputs)

x = logits[0, mask_token_index]
print(torch.log_softmax(x))
print(type(x))
print(logits[0, mask_token_index])

# target_id = inputs['input_ids'][mask_index]
# log_probs = torch.log_softmax(hs)[target_id]
# print(target_id)
# print(log_probs)


# # print(type(hs))

# # log_probs = log_softmax(hs)[target_id]
# # print(log_probs)
# #


# # predicted_token_class_ids = logits.argmax(-1)

# # # Note that tokens are classified rather then input words which means that
# # # there might be more predicted token classes than words.
# # # Multiple token classes might account for the same word
# # predicted_tokens_classes = [model.config.id2label[t.item()]
# #                             for t in predicted_token_class_ids[0]]
# # print(predicted_tokens_classes)
# # # ['Lead', 'Lead', 'Lead', 'Position', 'Lead', 'Lead',
# # #     'Lead', 'Lead', 'Lead', 'Lead', 'Lead', 'Lead']

# # labels = predicted_token_class_ids
# # loss = model(**inputs, labels=labels).loss
# # print(round(loss.item(), 2))


# # tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# # model = RobertaForMaskedLM.from_pretrained("roberta-base")

# # inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

# # with torch.no_grad():
# #     logits = model(**inputs).logits

# # # retrieve index of <mask>
# # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[
# #     0].nonzero(as_tuple=True)[0]

# # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
# # tokenizer.decode(predicted_token_id)
# # ' Paris'

# # labels = tokenizer("The capital of France is Paris.",
# #                    return_tensors="pt")["input_ids"]
# # # mask labels of non-<mask> tokens
# # labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

# # outputs = model(**inputs, labels=labels)
# # round(outputs.loss.item(), 2)
# # 0.1
# # # 0.25
