from operator import mod
import torch
import torch.utils.data as tdata
import csv
import os
import ftfy
import json
import nltk

def import_model_data_from_csv(filename='coded_data.csv'):
    """
    Import model training datset and output a list of JSON strings
    """
    # Setup data for input
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'data')
    coded_data = os.path.join(data_dir, 'coded-data.csv')


    # Read model training data into a list of dictionaries
    model_dicts = []
    with open(coded_data, 'r', encoding='utf-8') as c:
        reader = csv.DictReader(c)
        for line in reader:
            model_dicts.append(line)

    # Fix dodgy unicode encoding. Change field names to any 
    # string columns in your dataset you expect might contain ðŸŸ¢
    string_fields = ['Page name', 'Description', 'Page info']
    for line in model_dicts:
        for string_field in string_fields:
            line[string_field] = ftfy.fix_text(line[string_field])

    # Convert to JSON and return
    return json.loads(json.dumps(model_dicts))

def split_dataset(model_data, test_size=100, validate_size=100, seed=24):
    """
    Split a model dataset into training, test and validation sets

    return [training, test, validate]
    """
    #Set a generator with a fixed seed so we get the same sets every time
    gen = torch.Generator().manual_seed(seed)
    training_size = len(model_data) - (test_size + validate_size)
    return tdata.random_split(model_data, 
                              [training_size, test_size, validate_size],
                              generator=gen)



def main():
    model_data = import_model_data_from_csv()
    print(f"Total model length = {len(model_data)}")
    training, test, validate = split_dataset(model_data)
    pass


if __name__ == '__main__':
    main()

# Split data into training / validation  / test 
pass