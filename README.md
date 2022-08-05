A single convenience class, TorchModel, which uses PyTorch to convert .csv datasets into JSON training, test and validate data files.

e.g. - to create three JSON files containing a tokenised 'description' field for each record:
``` python
from torchtext.legacy import data

DESCRIPTION = data.Field(sequential=True, use_vocab = True, tokenize=your_favourite_tokeniser, lower=True)

w = TorchModel(model_csv = model_file,
               string_fields = ['description],
               conversions = field_conversions)

w.import_model_data_from_csv()
print(f"Records in model: {len(w.json_data)}")
w.split_dataset(test_size=1000, validate_size=100, seed=24)
```