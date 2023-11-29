import pandas as pd
import ast
intersentence_pd = pd.read_csv(
    'intersentence-formatted-data.csv')
intersentence_output_pd = pd.read_csv(
    'intersentence-formatted-data-model-output.csv')

intersentence_output_pd['gold_label'] = intersentence_output_pd['gold_label'].apply(
    ast.literal_eval)


options = [f'{i}' for i in range(3)]

intersentence_analyzed = pd.DataFrame(
    columns=['bias_type', 'target', 'model_output', 'result', 'model_order'])  # model_order refers to what number option was selected

data = list()
for index, rows in intersentence_pd.iterrows():
    model_output = intersentence_output_pd['model_answers'][index]
    output = None
    for option in options:
        if option in model_output:
            output = option
            break
    if output is None:
        for i in range(3):
            print(
                i, intersentence_output_pd['gold_label'][index], model_output)
            if str(intersentence_output_pd['gold_label'][index][i]) in model_output:
                output = i
    model_order = 4
    if output is not None:
        thingie = intersentence_output_pd['gold_label'][index]
        model_order = output
        output = intersentence_output_pd['gold_label'][index][int(output)]

    data.append(pd.Series([intersentence_output_pd['bias_type'][index],
                intersentence_output_pd['target'][index], model_output, output, model_order], index=intersentence_analyzed.columns))

intersentence_analyzed = pd.concat(data, axis=1).T
# intersentence_analyzed.concat(data, axis=1).T

intersentence_analyzed.to_csv('analyzed-intersentence.csv')
