import torch
import torch.nn as nn
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader


# Use the GPU if it's available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda") 
else:
    DEVICE = torch.device("cpu")

# Config
padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

# Perform a series of transforms in order
text_transform = T.Sequential(
    # Tokenise using Google's pretrained sentencePiece model
    T.SentencePieceTokenizer(xlmr_spm_model_path),

    # Encode tokens as numbers using the preset vocab
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    
    # Cut sequence to length
    T.Truncate(max_seq_len - 2),
    
    # Add a token to the start of the sequence
    T.AddToken(token=bos_idx, begin=True),
    
    # Add a token to the start of the sequence
    T.AddToken(token=eos_idx, begin=False),
)
