from distutils.debug import DEBUG
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

def swap_quotes(from_str):
    """
    Transfer single to double quotes and vice versa. Needed for JSON parsing
    """
    uniq = "~+][+~"
    return from_str.replace('"', uniq).replace("'", '"').replace(uniq, "'")

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

        # Use graphics card if it's available
        self.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
        )


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

        # Convert to JSON and return
        self.json_data = json.loads(json.dumps(model_dicts))

    ### TODO: Write a fn to process labels into numbers

    def define_torchtext_datasets(self):
        """
        Take training, test, validate JSON files and return a list of tt 
        datasets
        """
        self.train_df, self.valid_df, self.test_df = data.TabularDataset.splits(
            path= 'data',
            train = 'training.json',
            test = 'test.json',
            validation = 'validate.json',
            format = 'json',
            fields = self.conversions
        ) 

    def define_iterators(self):
        """
        Define an iterator which will return values from each dataset.
        """
        self.training, self.valid, self.test = data.BucketIterator.splits(
            (self.train_df, self.valid_df, self.test_df),
            batch_sizes = (4,4,4),
            device = self.device,
            sort = False
        )

    def build_vocabs(self):
        """
        Build vocabularies for each field in self.fields, based on vocab seen in
        training
        """
        for field in self.fields:
            field.build_vocab(self.train_df)
    
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

        # Write each set of lines to file        
        for dataset in 'training', 'test', 'validate':
            out_path = os.path.join(self.data_dir, f'{dataset}.json')
            with open(out_path, 'w', encoding='utf-8') as json_file:
                json_file.write(json.dumps(eval(dataset).dataset))

def main():
    print("This file really only instantiates a class, TorchModel - use that.")


if __name__ == '__main__':
    main()