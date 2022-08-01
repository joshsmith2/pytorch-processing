from operator import mod
import torch
import torch.utils.data as tdata
from torchtext.legacy import data
from torchtext.legacy import datasets
import nltk
from nltk.tokenize import word_tokenize

import csv
import os
import ftfy
import json

nltk.download('punkt')

class TorchModel:

    def __init__(self, model_csv, string_fields, conversions):
        # The name of the CSV file containing model data
        self.model_csv = model_csv
        
        # Any string fields in the data you want tokenising
        self.string_fields = string_fields

        # A list of tuples of the form ('FIELDNAME', 'Fieldname in CSV')
        self.conversions = conversions

        # Setup data for input
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(root_dir, 'data')
        self.coded_data = os.path.join(self.data_dir, self.model_csv)

    def import_model_data_from_csv(self):
        """
        Import model training datset and set self.json_data to a list of 
        JSON strings
        """

        # Read model training data into a list of dictionaries
        model_dicts = []
        with open(self.coded_data, 'r', encoding='utf-8') as c:
            reader = csv.DictReader(c)
            for line in reader:
                model_dicts.append(line)

        for line in model_dicts:
            for string_field in self.string_fields:
                # Fix dodgy unicode encoding. Change field names to any 
                # string columns in your dataset you expect might contain ðŸŸ¢
                line[string_field] = ftfy.fix_text(line[string_field])

                # Lowercase
                line[string_field] = line[string_field].lower()

                # Tokenise
                line[string_field] = word_tokenize(line[string_field])

        # Convert to JSON and return
        self.json_data = json.loads(json.dumps(model_dicts))

    def define_torchtext_datasets(self, training, test, validate):
        """
        Take three lists of tokenised JSON objects and return a list of tt 
        datasets
        """
        fields = {c[1]: (c[1], data.Field()) for c in self.conversions}
        
        #return data.TabularDataset.splits(
        #    path='data',
        #    train
        #) 
        
        # TODO: Write the json to .json files, I guess, so we can reload them here
    def split_dataset(self, 
                      test_size=100, validate_size=100, seed=24):
        """
        Split a model dataset into training, test and validation sets

        return [training, test, validate]
        """
        #Set a generator with a fixed seed so we get the same sets every time
        gen = torch.Generator().manual_seed(seed)
        training_size = len(self.json_data) - (test_size + validate_size)
        training, test, validate = tdata.random_split(self.json_data, 
                                                      [training_size, 
                                                       test_size, 
                                                       validate_size],
                                                      generator=gen)

        #TODO:Convert fieldnames?
        
        # Write each set of lines to file        
        for dataset in 'training', 'test', 'validate':
            out_path = os.path.join(self.data_dir, f'{dataset}.json')
            with open(out_path, 'w', encoding='utf-8') as json_file:
                json_file.write(str(eval(dataset).dataset))

        return training, test, validate

def main():
    print("This file really only instantiates a class, TorchModel - use that.")


if __name__ == '__main__':
    main()