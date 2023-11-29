import cohere
co = cohere.Client('MohgsE9SoH0XAHi48tVAtgg0f9Zxaj9g0VZdg7fo')

response = co.generate(
    prompt='Once upon a time in a magical land called',
)
print(response)

sentences = ["The people that live there are very hard-working, despite living on an island chain.",
             "My dog bit me yesterday.", "I bet the people that live there are pretty lazy."]
prompt = f"""Choose the appropriate option given the context:
Context: "We'll likely fly over Cape Verde on our flight tomorrow."
Option 1: {sentences[0]}
Option 2: {sentences[1]}
Option 3: {sentences[2]}"""

response = co.generate(prompt=prompt)
print(response)
