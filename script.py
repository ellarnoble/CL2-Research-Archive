import torch
import re
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np
import random
from tqdm import tqdm
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for randomness for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up logging system to record timestamped outputs in seperate text file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("coursework_results.txt"), 
        logging.StreamHandler()                        
    ]
)

""" Creating Datasets"""

# Collect training and validation data files from github repo 
# and collate into pandas df
base = Path("2024G2PST/data/tsv")
dfs = []

# Look for all .tsv files
for file_path in base.rglob("*.tsv"):
    parts = file_path.parts

    script_folder = parts[-4]
    lang_folder = parts[-3]
    
    # Choose selected scripts 
    if script_folder not in ("latin", "cyrillic"):
        continue

    # Skip files which contain mixed language data to avoid using duplicate data
    if lang_folder == script_folder:
        continue

    df = pd.read_csv(
        file_path, 
        sep="\t", 
        header=None, 
        names=["word", "ipa"],
        # prevent pandas from turning empty strings or "NaN" text into Null values
        keep_default_na=False,
        # ignore quote marks 
        quoting=3
    )

    df["script"] = script_folder
    df["language"] = lang_folder
    
    path_lowered = file_path.as_posix().lower()
    if "/train/" in path_lowered:
        df["split"] = "train"
    elif "/val/" in path_lowered:
        df["split"] = "val"
    else:
        df["split"] = "other"

    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

# Remove sylable boundary markers '.' as they are only present in indonesian language 
# and cause errors when conducting mixed language training and evaluation 
full_df['ipa'] = full_df['ipa'].str.replace('.', '', regex=False)

# Create training and validation dfs 
train_df = full_df[full_df["split"] == "train"].copy().reset_index(drop=True)
val_df = full_df[full_df["split"] == "val"].copy().reset_index(drop=True)

# Use 'task 2' folder from github repo to access test set data files 
test_base = Path("2024G2PST/data/eval/task_2")
test_dfs = []

# Exclude language files not seen in training (for this task)
# Exclude mixed language files to avoid duplicate test data
exclude_files = {"hbs_test.tsv", "hun_test.tsv"}
exclude_languages = {"latin", "abjad", "cyrillic"}

for file_path in test_base.rglob("*.tsv"):

    # Skip abjad folder (to align with selected training data)
    # (There is no devanagari folder in this directory)
    if "abjad" in file_path.parts:
        continue
    parts = file_path.parts
    file_name = file_path.name

    if file_name in exclude_files:
        continue

    lang_name = file_path.stem.replace("_test", "")
    if lang_name in exclude_languages:
        continue

    tdf = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["word", "ipa"],
        keep_default_na=False,
        quoting=3
    )

    tdf["script"] = parts[-2]
    tdf["language"] = lang_name
    tdf["split"] = "test"

    test_dfs.append(tdf)

test_df = pd.concat(test_dfs, ignore_index=True)

# Remove sylable boundary markers '.' to align with training and val data 
test_df['ipa'] = test_df['ipa'].str.replace('.', '', regex=False)

# Reset indexes in each df to start from 0
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

logging.info(f"Languages in Train Set: {sorted(full_df['language'].unique())}")
logging.info(f"Languages in Val Set: {sorted(val_df['language'].unique())}")
logging.info(f"Languages in Test Set: {sorted(test_df['language'].unique())}")

max_word_len = full_df['word'].astype(str).str.len().max()
max_ipa_len = full_df['ipa'].astype(str).str.split().str.len().max()

logging.info(f"Max Word Length: {max_word_len} graphemes")
logging.info(f"Max IPA Length:  {max_ipa_len} phonemes")

# Remove test exampels from training set 
test_words_set = set(test_df["word"].unique())
initial_train_size = len(train_df)
train_df = train_df[~train_df["word"].isin(test_words_set)].copy().reset_index(drop=True)

removed_count = initial_train_size - len(train_df)
logging.info(f"Removed {removed_count} samples from training set that were found in the test set.")
logging.info(f"New Training Set Size: {len(train_df)}")

# Initialise counters for source and target tokens
src_counter = Counter()
tgt_counter = Counter()

# Function for treating stress ' as a seperate token 
def tokenise_stress(text):
    text = re.sub(r"([ˈ])", r" \1 ", text)
    # Split IPA sequences into tokens 
    # Since strings are already seperated on white space this preserves 
    # phoneme level segmentation, e.g. 'tʃ')
    return text.split()

for _, row in train_df.iterrows():
    # Split each word into characters and count
    src_counter.update(list(row["word"]))
    # Count space seperated IPA symbols 
    tgt_counter.update(tokenise_stress(row["ipa"]))

# Give special tokens an index 
src_vocab = {"<pad>": 0, "<unk>": 1}
tgt_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

# Create language tokens
unique_languages = full_df["language"].unique()
lang_tokens = [f"<{lang}>" for lang in unique_languages]

# Assign language tokens to source tokens (with a fallback for unknown)
for i, lang_tag in enumerate(lang_tokens, start=2):
    src_vocab[lang_tag] = i

# Create script tokens 
unique_scripts = full_df["script"].unique()
script_tokens = [f"<{script}>" for script in unique_scripts]

# Add script tokens after language tokens -> len(src_vocab) 
# always equals next available index
start_idx = len(src_vocab)
for i, script_tag in enumerate(script_tokens, start=start_idx):
    src_vocab[script_tag] = i

# Add characters to source vocab
start_idx = len(src_vocab)
for i, tok in enumerate(sorted(src_counter.keys()), start=start_idx):
    src_vocab[tok] = i

# Add IPA tokens to target vocab (starting at 4 after special tokens)
for i, tok in enumerate(sorted(tgt_counter.keys()), start=4):
    tgt_vocab[tok] = i


class G2PDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab):
        self.df = df
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def encode_src(self, text, lang, script):
        script_tag = f"<{script}>"
        script_id = self.src_vocab.get(script_tag, self.src_vocab["<unk>"])
        
        lang_tag = f"<{lang}>"
        lang_id = self.src_vocab.get(lang_tag, self.src_vocab["<unk>"])
        
        char_ids = [self.src_vocab.get(c, self.src_vocab["<unk>"]) for c in text]
        return [script_id, lang_id] + char_ids

    def encode_tgt(self, text):
        tokens = tokenise_stress(text)
        ids = [self.tgt_vocab.get(t, self.tgt_vocab["<unk>"]) for t in tokens]
        return [self.tgt_vocab["<bos>"]] + ids + [self.tgt_vocab["<eos>"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "src": self.encode_src(row["word"], row["language"], row["script"]),
            "tgt": self.encode_tgt(row["ipa"])
        }

def prepare_batch(batch, pad_id=0):
    src_batch = [torch.tensor(x["src"]) for x in batch]
    tgt_batch = [torch.tensor(x["tgt"]) for x in batch]

    # Add padding to create sequences of equal length
    src = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    # Create a padding mask -> 1 = real token; 0 = pad token
    src_mask = (src != pad_id).int()
    tgt_mask = (tgt != pad_id).int()

    return {"src": src, "tgt": tgt, "src_mask": src_mask, "tgt_mask": tgt_mask}


train_dataset = G2PDataset(train_df, src_vocab, tgt_vocab)
val_dataset   = G2PDataset(val_df, src_vocab, tgt_vocab)
test_dataset   = G2PDataset(test_df, src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=prepare_batch)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=prepare_batch)
test_loader   = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=prepare_batch)


"""Model Architecture"""

class LexicalEmbedding(nn.Module):
    
    """A simple wrapper around nn.Embedding, which takes care to
    handle padding tokens """
    
    def __init__(self, vocab_size, input_dim, padding_idx):
        """
        Args:
            - vocab_size: the total number of unique tokens in the source vocabulary
            - input_dim: the size of each embedding vector 
            - padding_idx: the special index used for padding (0)
        """
        super().__init__()
        
        # nn.Embedding maps the token integers to embedding vectors
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_dim,
            padding_idx=padding_idx,
        )

    def forward(self, x: torch.Tensor):
        """
        This will accept both single and batched inputs.
        x.long casts the tensor to the correct integer type
        """
        return self.emb(x.long()) 

class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 64, padding_idx: int = 0):
        """
        Args:
            d_model: hidden size
            max_len: max sequence length supported (64 is sufficient for max sequence length observed in dataset)
            padding_idx: index in input_ids used for padding (0)
        """
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx

        self.position_embeddings = nn.Embedding(max_len, d_model, padding_idx=0)
        
        
        # Pad positions are mapped to index 0 in the position table.
        # (i.e., row 0 stays zeroed and is never updated)
        
        with torch.no_grad():
            self.position_embeddings.weight[0].zero_()

    
    # Use mask to zero out padding tokens 
    # add positional embeddings to lexical embeddings
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        position_ids = (torch.cumsum(mask, dim=1) * mask).to(device).to(torch.long)

        pos_emb = self.position_embeddings(position_ids)
        return x + pos_emb

class MultiHeadAttention(nn.Module):
    """"A multi head attention module which can be adapted for cross attention 
    Q, K, and V are projected once and then split across multiple heads"""

    def __init__(self, input_dim, num_heads):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        """
        Args:
                input_dim: embedding size
                num_heads: number of attention heads
        """
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads


        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)

        # Combines all heads back into a single representation
        self.out_projection = nn.Linear(input_dim, input_dim)

    def forward(self, X, context = None, mask = None, causal_masking=False):
        batch_size, seq_len, _ = X.shape

        # Allows module to be used for self attention and cross attention
        kv_input = context if context is not None else X
        seq_len_kv = kv_input.shape[1]

        # Generate queries, keys and values using learnable linear projections
        Q = self.query_projection(X)
        K = self.key_projection(kv_input)
        V = self.value_projection(kv_input)

        """
        Split Q, K, V into a different view for each attention head
        Dims: 
        Q: (batch_size, num_heads, seq_len, head_dim)
        K,V: (batch, heads, seq_len_kv, head_dim)
        """
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)     
        
    
        # Calculate attention scores:
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply padding mask if needed
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) 
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply causal masking if set to True
        if causal_masking:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        # Calculate attention weights using softmax over scores and combine with values
        attention_weights = torch.softmax(scores, dim=-1)
        context = attention_weights @ V
        
        """
        Dims transformation:

        Initial:
        (batch_size, num_heads, seq_len, head_dim)

        1. Move sequence dimension before heads
        → (batch_size, seq_len, num_heads, head_dim)

        2. Concatenate attention heads (num_heads × head_dim)
        → (batch_size, seq_len, input_dim)

        3. Apply learned output projection to mix information across heads
        """
        context = context.transpose(1, 2)
        context = context.reshape(batch_size, seq_len, self.input_dim)
        return self.out_projection(context)

class LayerNorm(nn.Module):
    """BERT-style LayerNorm: normalise over last dimension with learnable
    scale (gamma) and bias (beta) """
    
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
            torch.var(x, dim=-1, correction=0, keepdim=True)  + self.eps
        ) * self.weight + self.bias

class MLP(nn.Module):
    """Two-layer MLP, applied elementwise, over the last dimension,
    using GELU as the activation function"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fully_connected1 = nn.Linear(input_dim, hidden_dim)
        self.fully_connected2 = nn.Linear(hidden_dim, input_dim)
        self.gelu = nn.GELU()
        self.everything = nn.Sequential(
            self.fully_connected1,
            self.gelu,
            self.fully_connected2
        )

    def forward(self, X):
        return self.everything(X)

class EncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_p = 0.1):
        super().__init__()

        # Attention block with dropout
        self.norm1 = LayerNorm(input_dim)
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout_p)

        # MLP block with dropout
        self.norm2 = LayerNorm(input_dim)
        self.mlp = MLP(input_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_p)

    # Causal masking set to False for encoder design
    def forward(self, X, mask=None, causal_masking=False):
        attn_out = self.attention(
            self.norm1(X),
            mask=mask,
            causal_masking=causal_masking
        )
        X = X + self.dropout1(attn_out)

        mlp_out = self.mlp(self.norm2(X))
        X = X + self.dropout2(mlp_out)

        return X

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_p=0.1):
        super().__init__()

        
        self.norm1 = LayerNorm(input_dim)
        self.self_attn = MultiHeadAttention(input_dim, num_heads)

        
        self.norm2 = LayerNorm(input_dim)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads)
        
        self.norm3 = LayerNorm(input_dim)
        self.mlp = MLP(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked self attention block
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, mask = tgt_mask, causal_masking=True)
        x = x + self.dropout(attn_out)
        
        # Cross attention block -> Query: decoder; Key + Value: encoder output
        norm_x = self.norm2(x)
        attn_out = self.cross_attn(norm_x, context = enc_output, mask = src_mask)
        x = x + self.dropout(attn_out)

        # MLP block
        norm_x = self.norm3(x)
        x = x + self.dropout(self.mlp(norm_x))

        return x


class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        dropout_p=0.1
    ):
        super().__init__()
        """Implement encoder and decoder modules and 
        combine with lexical and position embeddings for full model"""

        self.src_lexical = LexicalEmbedding(src_vocab_size, input_dim, padding_idx=0)
        self.tgt_lexical = LexicalEmbedding(tgt_vocab_size, input_dim, padding_idx=0)
        self.pos_emb = LearnedPositionalEmbedding(d_model=input_dim, max_len=64)

        self.encoder_stack = nn.ModuleList([
            EncoderLayer(input_dim, num_heads, hidden_dim, dropout_p) 
            for _ in range(num_layers)
        ])
        
        self.decoder_stack = nn.ModuleList([
            DecoderLayer(input_dim, num_heads, hidden_dim, dropout_p) 
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(input_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Create representation of the source input (characters) using lexical and positional embeddings
        Pass these representations through stacked encoder layers 
        """
        src_emb = self.src_lexical(src)
        enc_x = self.dropout(self.pos_emb(src_emb, src_mask))
        for layer in self.encoder_stack:
            enc_x = layer(enc_x, src_mask)
            
        """
        Create representation of the target input (symbols) using lexical and positional embeddings
        Pass these representations through stacked decoder layers in which each layer performs 
        self-attention on previously generated symbols and cross-attention on the encoder's character representations
        """
        tgt_emb = self.tgt_lexical(tgt)
        dec_x = self.dropout(self.pos_emb(tgt_emb, tgt_mask))
        for layer in self.decoder_stack:
            dec_x = layer(dec_x, enc_x, src_mask, tgt_mask)
            
        return self.fc_out(dec_x)
    
"""Training functions"""

def compute_loss(logits, target_ids, vocab_size, loss_fn):
    # Remove first token (<bos>) from each target sequence
    target_labels = target_ids[:, 1:]
    # Flatten logits (remove batch dimension) for pytorch loss function
    logits = logits.reshape(-1, vocab_size)
    # Flatten (batch, seq_len) → (batch * seq_len) 
    # to match reshaped logits for token-level loss
    target_labels = target_labels.reshape(-1)
    return loss_fn(logits, target_labels)


def train_step(model, batch, optimizer, vocab_size, loss_fn, device):
    model.train()
    optimizer.zero_grad()

    # Move all tensors to device 
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)
    src_mask = batch["src_mask"].to(device)
    tgt_mask = batch["tgt_mask"].to(device)

    # Forwards pass with teacher forcing 
    logits = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1])

    # # Backpropagate loss and update parameters
    loss = compute_loss(logits, tgt, vocab_size, loss_fn)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # Return scalar loss for logging
    return loss.item()


def train_epoch(model, dataloader, optimizer, vocab_size, loss_fn, device, no_progress_bar=False):
    """Use train step function to run a forwards pass over training dataset
    (one epoch) and return average loss"""

    total_loss = 0.0
    for batch in tqdm(dataloader, disable=no_progress_bar, leave=False):
        loss = train_step(model, batch, optimizer, vocab_size, loss_fn, device)
        total_loss += loss
    return total_loss / len(dataloader)

def levenshtein_distance(ref, hyp):
    n, m = len(ref), len(hyp)

    # Create table: [(m+1) x (n+1)]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1

            # Consider deletions, insertions and subsitutions
            dp[i][j] = min(
                dp[i - 1][j] + 1,        
                dp[i][j - 1] + 1,        
                dp[i - 1][j - 1] + cost  
            )

    # Return raw distance 
    return dp[n][m]


def levenshtein_stats(ref, hyp):
    n, m = len(ref), len(hyp)
    """More comprhensive function for reporting counts of D, I, S"""
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        
                dp[i][j - 1] + 1,        
                dp[i - 1][j - 1] + cost  
            )

    # Count S, I, D
    i, j = n, m
    S = I = D = 0

    while i > 0 or j > 0:
        # Match or substitution
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 1:
                    S += 1
                i -= 1
                j -= 1
                continue

        # Deletion
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
            continue

        # Insertion
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
            continue

    distance = dp[n][m]

    # Normalised PER
    if n == 0:
        per = 0.0 if m == 0 else 1.0
    else:
        per = distance / n

    return {
        "S": S,
        "I": I,
        "D": D,
        "raw": distance,
        "PER": per
    }

# Greedy generate function for evaluation of PER on validation and test set 
def greedy_generate(model, src, src_mask, prefix, max_len, eos_idx):
    model.eval()

    # Add batch dimension 
    result = prefix.unsqueeze(0)
    src = src.unsqueeze(0)
    src_mask = src_mask.unsqueeze(0)

    with torch.no_grad():
        # Loop until maximum sequence length is reached
        for _ in range(max_len - result.size(1)):
            
            # Initialise a target mask
            tgt_mask = torch.ones_like(result).to(result.device)
            logits = model(src, result, src_mask, tgt_mask)

            # Greedy decoding step: chooses token with highest probability
            next_token = logits[:, -1, :].argmax(-1)
            # Append predicted token to sequence and stop generation if eos token is produced
            result = torch.cat([result, next_token.view(1, 1)], dim=1)
            if next_token.item() == eos_idx:
                break

    # Remove the batch dimension and return generated sequence 
    return result.squeeze(0)

def validate_epoch(model, dataloader, loss_fn, device, pad_token_id=0, no_progress_bar=False):
    model.eval()

    total_loss = 0.0
    total_raw = 0
    total_N = 0

    correct = 0 
    total = 0   

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=no_progress_bar, leave=False):

            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)

            logits = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1])
            loss = compute_loss(logits, tgt, logits.size(-1), loss_fn)
            total_loss += loss.item()

            for i in range(src.size(0)):
                pred_ids = greedy_generate(
                    model,
                    src[i],
                    src_mask[i],
                    prefix=torch.tensor([tgt_vocab['<bos>']]).to(device),
                    max_len=50,
                    eos_idx=tgt_vocab['<eos>']
                )

                special_tokens = {
                    tgt_vocab['<bos>'],
                    tgt_vocab['<eos>'],
                    tgt_vocab['<pad>'],
                    tgt_vocab['<unk>']
                }

                pred_tokens = [t.item() for t in pred_ids if t.item() not in special_tokens]
                gold_tokens = [t.item() for t in tgt[i] if t.item() not in special_tokens]

                # PER
                total_raw += levenshtein_distance(gold_tokens, pred_tokens)
                total_N += len(gold_tokens)

                # Word Accuracy 
                if pred_tokens == gold_tokens:
                    correct += 1
                total += 1

    per = total_raw / total_N if total_N > 0 else 0.0
    word_acc = correct / total if total > 0 else 0.0

    return {
        "loss": total_loss / len(dataloader),
        "per": per,
        "word_acc": word_acc 
    }


"""Model Initialisation """
# Set model hyperparameters
INPUT_DIM = 512
HIDDEN_DIM = 2048
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.2

# Get vocab sizes from dictionaries
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# Initialise the model
model = EncoderDecoderModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout_p=DROPOUT
)

model.to(device)

logging.info(f"Model initialized with {src_vocab_size} source tokens and {tgt_vocab_size} target tokens.")

lang_indices = [src_vocab[tok] for tok in lang_tokens]
initial_lang_embs = model.src_lexical.emb.weight[lang_indices].clone().detach()

# Initialize loss and optimizer
# Label smoothing to reduce model overconfidence 
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr = 1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# Set up early stopping logic using validation PER
patience = 10   
patience_counter = 0     
best_val_per = float('inf')
best_epoch = 0
delta = 0.0001

# Intialise metrics for plotting 
train_losses = []
val_losses = []
val_pers = []
val_word_accs = []

""" Model Training """
num_epochs = 100
logging.info("Model training starting...")
logging.info(f"Running on: {device}")
logging.info("-" * 30)

for epoch in range(num_epochs):

    # Forwards pass through model 
    avg_train_loss = train_epoch(
        model, train_loader, optimizer, tgt_vocab_size, criterion, device
    )

    # Validate for monitoring and early stopping 
    val_metrics = validate_epoch(
        model, val_loader, criterion, device, pad_token_id=0
    )
    
    current_val_loss = val_metrics["loss"]
    current_val_per = val_metrics["per"]
    current_val_word_acc = val_metrics["word_acc"]
    
    scheduler.step(current_val_loss)

    logging.info(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"| Train Loss: {avg_train_loss:.4f} "
        f"| Val Loss: {current_val_loss:.4f} "
        f"| Val PER: {current_val_per*100:.2f}%"
        f"| Val Word Acc: {current_val_word_acc*100:.2f}%"
)


    train_losses.append(avg_train_loss)
    val_losses.append(current_val_loss)
    val_pers.append(current_val_per)
    val_word_accs.append(current_val_word_acc)

    # Early stopping logic 
    if current_val_per < best_val_per - delta:
        best_val_per = current_val_per
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        logging.info("-" * 30)
        logging.info(f"Early stopping triggered after {patience} epochs of no significant PER improvement.")
        logging.info("Best model saved.")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()

# Plot training and val loss; val PER and word accuracy over epochs 
def plot_training_curves(
    train_losses,
    val_losses,
    val_pers,
    val_word_accs,
    best_epoch,
    save_path="training_curves.png"
):
    epochs = range(1, best_epoch + 1)

    # Slice to best epoch
    train_losses = np.array(train_losses[:best_epoch])
    val_losses = np.array(val_losses[:best_epoch])
    val_pers = np.array(val_pers[:best_epoch])
    val_word_accs = np.array(val_word_accs[:best_epoch])

    # Create side by side subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    axes[0].plot(epochs, train_losses, label="Training Loss")
    axes[0].plot(epochs, val_losses, label="Validation Loss")
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].legend(fontsize=14)
    axes[0].tick_params(labelsize=14)
    axes[0].grid()

    axes[1].plot(epochs, val_pers, label="Validation PER")
    val_word_error = 1 - val_word_accs
    axes[1].plot(epochs, val_word_error, label="Validation WER")
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Score", fontsize=14)
    axes[1].legend(fontsize=14)
    axes[1].tick_params(labelsize=14)
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    val_pers=val_pers,
    val_word_accs=val_word_accs,
    best_epoch=best_epoch
)

""" Model Evaluation """

def calculate_eval_metrics(model, data_loader, tgt_vocab, max_len, eos_idx, device):
    model.eval()

    # Create a reverse vocab; converts token ids into tokens
    inv_vocab = {v: k for k, v in tgt_vocab.items()}

    # Initialise metrics for calculating PER
    total_distance = 0
    total_ref_len = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:

            src_batch = batch["src"].to(device)
            mask_batch = batch["src_mask"].to(device)
            tgt_batch = batch["tgt"]

            # Iterate through each example in the batch and 
            # generate output sequence token by token using greedy decoding
            # Stop at eos token or max length
            for i in range(src_batch.size(0)):

                pred_ids = greedy_generate(
                    model,
                    src_batch[i],
                    mask_batch[i],
                    prefix=torch.tensor([tgt_vocab['<bos>']]).to(device),
                    max_len=max_len,
                    eos_idx=eos_idx
                )

                special_tokens = {'<bos>', '<eos>', '<pad>', '<unk>'}

                # Convert predicted ids into tokens, filtering out special tokens
                pred_tokens = [
                    inv_vocab.get(idx.item(), "<unk>")
                    for idx in pred_ids
                    if inv_vocab.get(idx.item(), "<unk>") not in special_tokens
                ]

                # Do the same process for the gold labels 
                target_tokens = [
                    inv_vocab.get(idx.item(), "<unk>")
                    for idx in tgt_batch[i]
                    if inv_vocab.get(idx.item(), "<unk>") not in special_tokens
                ]

                # Store full sequences for calculating word accuracy
                all_preds.append(pred_tokens)
                all_targets.append(target_tokens)

                # Computate edit distance between and gold phoneme sequence
                total_distance += levenshtein_distance(target_tokens, pred_tokens)
                total_ref_len += len(target_tokens)

    # Normalise edit distance for PER (micro-averaged)
    per = total_distance / total_ref_len if total_ref_len > 0 else 0.0

    # Compute word accuracy
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    total = len(all_preds)
    exact_match_accuracy = correct / total if total > 0 else 0.0

    return {
        "word_acc": exact_match_accuracy,
        "PER": per,
        "all_preds": all_preds,
        "all_targets": all_targets
    }

test_metrics = calculate_eval_metrics(
    model, 
    test_loader, 
    tgt_vocab, 
    max_len=50, 
    eos_idx=tgt_vocab['<eos>'], 
    device=device
)
word_acc = test_metrics["word_acc"]
per = test_metrics["PER"]
all_preds = test_metrics["all_preds"]
all_targets = test_metrics["all_targets"]

logging.info(
    f"Overall Word Accuracy: {word_acc:.2%} | "
    f"Overall PER: {per:.2%}"
)

total_S, total_I, total_D = 0, 0, 0
for pred, gold in zip(all_preds, all_targets):
    stats = levenshtein_stats(gold, pred)
    total_S += stats["S"]
    total_I += stats["I"]
    total_D += stats["D"]

logging.info(f"Total Substitutions: {total_S}")
logging.info(f"Total Insertions: {total_I}")
logging.info(f"Total Deletions: {total_D}")
logging.info(f"Substitution proportion: {total_S / (total_S + total_I + total_D):.2%}")

def get_grouped_dataloaders(df, group_by_col, src_vocab, tgt_vocab, batch_size=32):
    """
    Creates a dictionary of DataLoaders grouped by a specific column.
    group_by_col: "language" or "script"
    """
    grouped_loaders = {}
    
    for group_name, group_df in df.groupby(group_by_col):
    
        subset_dataset = G2PDataset(group_df, src_vocab, tgt_vocab)
        
        loader = DataLoader(
            subset_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=prepare_batch
        )
        
        grouped_loaders[group_name] = loader
        
    return grouped_loaders

lang_loaders = get_grouped_dataloaders(test_df, "language", src_vocab, tgt_vocab)

# Report evaluation metrics per language 

lang_metrics = {}
for lang, loader in lang_loaders.items():
    lang_metrics[lang] = calculate_eval_metrics(
        model=model,
        data_loader=loader,
        tgt_vocab=tgt_vocab,
        max_len=50,
        eos_idx=tgt_vocab['<eos>'],
        device=device
    )
    logging.info(
        f"Result for {lang} - "
        f"Word Acc: {lang_metrics[lang]['word_acc']:.2%}, "
        f"PER: {lang_metrics[lang]['PER']:.2%}"
    )

""" Analysis of language token embeddings """

logging.info("Analysing language token embeddings")
# Get embedding matrix and build language token list
emb_matrix = model.src_lexical.emb.weight.detach().cpu()

# Slice embedding matrix and normalize embeddings
lang_embs = emb_matrix[lang_indices]
lang_embs = F.normalize(lang_embs, p=2, dim=1)

# Cosine similarity matrix
sim_matrix = lang_embs @ lang_embs.T 

sim_df = pd.DataFrame(
    sim_matrix.numpy(),
    index=lang_tokens,
    columns=lang_tokens
)

# Create plot
plt.figure(figsize=(8, 7))
plt.imshow(sim_df.values)
plt.colorbar()

plt.xticks(range(len(lang_tokens)), lang_tokens, rotation=90, fontsize=14)
plt.yticks(range(len(lang_tokens)), lang_tokens, size=14)

plt.tight_layout()
plt.savefig("lang_embedding_similarity.png", dpi=300, bbox_inches="tight")
logging.info("Language Token Embedding Plot saved to lang_embedding_similarity.png")
plt.close()

""" Error Analysis """

# Subsitution error analysis (most common substitutions)
def collect_substitution_errors(all_preds, all_targets):
    errors = Counter()

    for pred, gold in zip(all_preds, all_targets):
        for p, g in zip(pred, gold):
            if p != g:
                errors[(g, p)] += 1 

    return errors

errors = collect_substitution_errors(all_preds, all_targets)
logging.info("Top 20 Most Common Substitution Errors")
logging.info(errors.most_common(20))
logging.info("-" * 30)

# Substitutions per language 
def lang_substitution_errors(all_preds, all_targets, lang, n=10):
    errors = Counter()
    for pred, gold in zip(all_preds, all_targets):
        for p, g in zip(pred, gold):
            if p != g:
                errors[(g, p)] += 1
    
    logging.info(f"Top {n} substitution errors for {lang}:")
    logging.info(errors.most_common(n))
    logging.info("-" * 30)

for lang, metrics in lang_metrics.items():
    lang_substitution_errors(metrics["all_preds"], metrics["all_targets"], lang)

# Length error analysis (over or underproduction)
def length_analysis(all_preds, all_targets):
    diffs = []

    for pred, gold in zip(all_preds, all_targets):
        diffs.append(len(pred) - len(gold))

    return {
        "avg_diff": sum(diffs) / len(diffs),
        "max_over": max(diffs),
        "max_under": min(diffs)
    }

length_stats = length_analysis(
    test_metrics["all_preds"],
    test_metrics["all_targets"]
)
logging.info("Error Analysis of Over and Under Production") 
logging.info(length_stats)
logging.info("-" * 30)

# Get a few bad examples per language
def bad_examples(all_preds, all_targets, n=3):
    examples = []

    for pred, gold in zip(all_preds, all_targets):
        dist = levenshtein_distance(gold, pred)
        examples.append((dist, gold, pred))

    examples.sort(reverse=True)

    for d, g, p in examples[:n]:
        logging.info(f"Distance: {d}")
        logging.info(f"Gold: {g}")
        logging.info(f"Pred: {p}")
        logging.info("-" * 30)

for lang, metrics in lang_metrics.items():
    logging.info(f"Bad examples for {lang}:")
    bad_examples(metrics["all_preds"], metrics["all_targets"], n=3)

# Calculate number of substitutions/insertions and deletions per languages
for lang, metrics in lang_metrics.items():
    lang_preds = metrics["all_preds"]
    lang_targets = metrics["all_targets"]
    total_S, total_I, total_D = 0, 0, 0
    for pred, gold in zip(lang_preds, lang_targets):
        stats = levenshtein_stats(gold, pred)
        total_S += stats["S"]
        total_I += stats["I"]
        total_D += stats["D"]
    total = total_S + total_I + total_D
    if total > 0:
        logging.info(
            f"{lang} - "
            f"S: {total_S} ({total_S/total:.1%}), "
            f"I: {total_I} ({total_I/total:.1%}), "
            f"D: {total_D} ({total_D/total:.1%})"
        )