import pandas as pd
import ast
intrasentence_pd = pd.read_csv(
    'intrasentence-formatted-data.csv')
intrasentence_output_pd = pd.read_csv(
    'intrasentence-formatted-data-model-output.csv')

intrasentence_output_pd['gold_label'] = intrasentence_output_pd['gold_label'].apply(
    ast.literal_eval)


options = [f'{i}' for i in range(3)]

intrasentence_analyzed = pd.DataFrame(
    columns=['bias_type', 'target', 'model_output', 'result', 'model_order'])  # model_order refers to what number option was selected

data = list()
for index, rows in intrasentence_pd.iterrows():
    model_output = intrasentence_output_pd['model_answers'][index]
    output = None
    for option in options:
        if option in model_output:
            output = option
            break
    if output is None:
        for i in range(3):
            print(
                i, intrasentence_output_pd['gold_label'][index], model_output)
            if str(intrasentence_output_pd['gold_label'][index][i]) in model_output:
                output = i
    model_order = 4
    if output is not None:
        thingie = intrasentence_output_pd['gold_label'][index]
        model_order = output
        output = intrasentence_output_pd['gold_label'][index][int(output)]

    data.append(pd.Series([intrasentence_output_pd['bias_type'][index],
                intrasentence_output_pd['target'][index], model_output, output, model_order], index=intrasentence_analyzed.columns))

intrasentence_analyzed = pd.concat(data, axis=1).T
# intrasentence_analyzed.concat(data, axis=1).T

intrasentence_analyzed.to_csv('analyzed-intrasentence.csv')
