from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import json
import numpy as np
import math
# from transformers import BertTokenizer
from transformers import GPT2Tokenizer
# from tokenizers import Tokenizer, processors


import pandas as pd

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import time
import os
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler

import matplotlib.pyplot as plt




scaler = GradScaler()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)



import torch
import torch.nn as nn
import torch.nn.functional as F


# model architecture
class AttentionHead(nn.Module):
  """a single head of self attention"""
  
  def __init__(self, n_embed, head_size, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    B, T, C = x.shape
    K = self.key(x) # (B, T, C)
    Q = self.query(x) # (B, T, C)
    
    wei = Q @ K.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, H, C) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    V = self.value(x) # (B, T, C)
    out = wei @ V # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out
  
class MultiHeadAttention(nn.Module):
  """a multi-head self attention layer"""
  
  def __init__(self, n_embed, n_heads, head_size, block_size, dropout):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(n_embed, head_size, block_size, dropout) for _ in range(n_heads)])
    self.fc = nn.Linear(head_size * n_heads, n_embed)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_heads*C)    
    out = self.fc(out) # (B, T, C)
    out = self.dropout(out) 
    return out
  
class FeedForward(nn.Module):
  def __init__(self, n_embed, n_hidden, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_embed),
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
   return self.net(x)
  
class Block(nn.Module):
  def __init__(self, n_embed, n_heads, block_size, dropout):
    super().__init__()
    self.sa_heads = MultiHeadAttention(n_embed, n_heads, n_embed // n_heads, block_size, dropout)
    self.ffwd = FeedForward(n_embed, n_embed*4, dropout)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
    
    
  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x)) #  [batch_size, block_size, n_embed]
    x = x + self.ffwd(self.ln2(x)) # [batch_size, block_size, n_embed]
    return x

class Transformer_Model(nn.Module):
  def __init__(self, vocab_size, block_size, n_embed, n_heads, n_layers, dropout, device="cpu"):
    super().__init__()

    print(vocab_size, block_size, n_embed, n_heads, n_layers, dropout)
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
    
    self.device = device
    self.block_size = block_size

  def forward(self, idx, targets=None, pad_token_id=0):
    # idx and target are both [batch_size, block_size]
    B, T = idx.shape
    
    tok_emb = self.token_embedding_table(idx) # [batch_size, block_size, n_embed]
    pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # [block_size, n_embed]
    x = tok_emb + pos_emb # [batch_size, block_size, n_embed]
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # [batch_size, block_size, vocab_size]
    
    if targets is None:
        loss = None
        n_tokens = 0
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.reshape(B*T)
        
        # Create a mask for non-padding tokens
        non_pad_mask = (targets != pad_token_id)
        
        # Apply the mask to both logits and targets
        logits = logits[non_pad_mask]
        targets = targets[non_pad_mask]
        
        # Calculate the number of non-padding tokens
        n_tokens = non_pad_mask.sum().item()
        
        loss = F.cross_entropy(logits, targets)
    
    return logits, loss, n_tokens
      
  def forward_old(self, idx, targets=None):
      # idx and target are both [batch_size, block_size]
      B, T = idx.shape
      
      tok_emb = self.token_embedding_table(idx) # [batch_size, block_size, n_embed]
      pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # [block_size, n_embed]
      x = tok_emb + pos_emb # [batch_size, block_size, n_embed]
      x = self.blocks(x)
      x = self.ln_f(x)
      logits = self.lm_head(x) # [batch_size, block_size, vocab_size]
      
      if targets is None:
        loss = None
      else:

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.reshape(B*T)
        loss = F.cross_entropy(logits, targets)
      
      return logits, loss
      
  def generate(self, idx, max_new_tokens=100):
    # idx is (B, T)
    for _ in range(max_new_tokens):
      # get the last block_size tokens
      idx_cond = idx[:, -self.block_size:] # (B, T)
      # get the predictions
      logits, _ = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      
    return idx



# gen_text_time = 60
# max_seq_len = 512#128#5000 #512
# max_seq_len_given = 129
# save_dir = "151m_tiny_stories"
# batch_size = 24

gen_text_time = 60
max_seq_len = 1024#128#5000 #512
max_seq_len_given = 256+1
save_dir = "70m_wikipedia"
batch_size = 28
checkpoint_save_freq = 1000
lr = 0.2e-3

effective_batch_size = 48

tokens_per_batch = (max_seq_len_given-1) * effective_batch_size
print("tokens per batch", tokens_per_batch)



class MessageDataset_parquet(Dataset):
    def __init__(self, max_count=-1, batch_size=16):

        self.pretokenized = True
        self.q_and_a = False

        self.batch_size = batch_size
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # self.tokenizer.enable_padding(pad_id=0, pad_token="<|im_end|>", length=max_seq_len_given)
        # self.tokenizer.enable_truncation(max_length=max_seq_len_given)
        # tokenizer = Tokenizer(BPE())
        # self.tokenizer = Tokenizer.from_file("/media/nathans/AE8D-87D2/llm_big/slms-master/data/parquet_tokenizer_lower2.json")

        

        # self.tokenizer = Tokenizer.from_file("/media/nathans/AE8D-87D2/llm_big/slms-master/data/parquet_tokenizer_lower2.json")
        # self.tokenizer.post_processor = SimplePostProcessor()

        # string = "Hello, how are you? This is a missspelled word".lower()
        # tokenized_data = self.tokenizer.encode(string)
        # print(self.tokenizer.decode(tokenized_data, skip_special_tokens=True))

        self.pth = "fineweb_parquet_pretokenized"
        self.pth = "bookcorpus_parquet"
        self.pth = "wikipedia_parquet"
        self.pth = "wikipedia_parquet_tokenized"
        # self.pth = "fineweb_parquet_clean"
        # self.pth = "tiny_stories"
        ds = os.listdir(self.pth)
        self.datasets = []
        for d in ds:
            if d[-5::] == "rquet":
                self.datasets.append(d)



        # print("TRUNCATING DATASETS")
        # self.datasets = self.datasets[0:8]


        self.datasets.sort()
        # self.datasets.reverse()
        print(self.datasets)
        self.dataset_ind = 7#int(np.random.random() * (len(self.datasets)))-1
        vocabulary = self.tokenizer.get_vocab()
        # print(vocabulary)
        self.num_words = len(vocabulary)
        print("vocab size", self.num_words)
        self.tokens_per_pq = 302e6

        


    def load_dataset(self):
        self.loaded = 0
        
        self.dataset_ind = (self.dataset_ind + 1) % len(self.datasets)

        
        n = self.datasets[self.dataset_ind]

        print("loading dataset", n)
        df = pd.read_parquet(os.path.join(self.pth, n), engine='pyarrow')

        print(df)
        if self.pretokenized:
            df.dropna(subset=['tokenized'], inplace=True)

            convs = df.reset_index(drop=True)
            tokens_used = 0
            self.conversations = []
            print("loaded. re-arranging")
            for c in convs['tokenized']:
                # print(c)
                while len(c) > max_seq_len_given:
                    self.conversations.append(c[0:max_seq_len_given])
                    c = c[max_seq_len_given::]
                    tokens_used += max_seq_len_given
                if len(c) > max_seq_len_given / 2:
                    self.conversations.append(c)
                    tokens_used += len(c)
            print("total tokens", int(tokens_used/1e6), "m")
        else:
            print("loaded. sorting now")

            if self.q_and_a:

                self.conversations = df


            else:

                df['text_length'] = df['text'].apply(len)
                df_sorted = df.sort_values(by='text_length')
                df_sorted = df_sorted.drop(columns=['text_length'])
                
                lens = [len(a) for a in df["text"]]
                print("avg", sum(lens)/len(lens), max(lens), min(lens))

                df_sorted['text_length'] = df_sorted['text'].apply(len)
                df_filtered = df_sorted[(df_sorted['text_length'] >= 100)]# & (df_sorted['text_length'] <= 3000)]
                df_filtered = df_filtered.drop(columns=['text_length'])
                # df_filtered = df_filtered.reset_index(drop=True)


                self.conversations = df_filtered.reset_index(drop=True)


        print("loaded dataset.", len(self.conversations), "conversations")



    def __len__(self):
        return len(self.conversations)


    
    def __getitem__(self, idx):
        if self.pretokenized:
            inputs = torch.tensor(self.conversations[idx])

        else:


            if self.q_and_a:

                t = self.conversations["prompt"][idx]

                inputs = self.tokenizer.encode(t, max_length=int(max_seq_len_given*0.8), truncation=True)

                inputs += [0]
                t = self.conversations["response"][idx]
                inputs += self.tokenizer.encode(t, max_length=int(max_seq_len_given*0.8), truncation=True)
                inputs = torch.tensor(inputs)

            else:

                t = self.conversations["text"][idx]
                
                t = t.replace("a. m.", "AM")
                t = t.replace("p. m.", "PM")

                t = t.replace('"', "")
                t = t.replace(" '", "")
                t = t.replace("' ", "")
                t = t.replace("--", "")
                # print("t->", t)
                inputs = torch.tensor(self.tokenizer.encode(t, max_length=max_seq_len_given, truncation=True))

        s = inputs.size(0)            
        # print("was", inputs.shape)
        # if inputs.size(0) == max_seq_len:
        #     print(len(self.tokenizer.decode(inputs, skip_special_tokens=True)))
        if inputs.size(0) > max_seq_len_given:
            inputs = inputs[0:max_seq_len_given]
        padding_length = max_seq_len_given - inputs.size(0)

        if padding_length > 0:
            inputs = torch.nn.functional.pad(inputs, (0, padding_length), 'constant', 0)

        # print(inputs)
        # if torch.sum(inputs==1) + torch.sum(inputs==366) + torch.sum(inputs==1377) + torch.sum(inputs==438) > 0:
        #     print("woah", inputs, torch.sum(inputs==0), torch.sum(inputs==366), inputs==366)
        #     print("woah", t)

        # print(inputs.shape)

        return inputs, s


plt.ion()
line = None
fig = None
ax = None
def make_plot(vals):
    global line
    global fig
    global ax
    x = range(len(vals))
    y = vals
    fig, ax = plt.subplots()
    line, = ax.plot(x, y)

    plt.show()
    line.set_ydata(y)
    line.set_xdata(x)
    fig.canvas.draw()
    fig.canvas.flush_events()


def update_plot(vals):
    global line
    global fig
    global ax
    x = list(range(len(vals)))
    y = vals
    line.set_ydata(y)
    line.set_xdata(x)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()



def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, device='cuda'):
    model.eval()
    generated = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # tokenized_data = tokenizer.encode(prompt)
    # generated = torch.tensor(tokenized_data.ids).to(device)
    # generated = generated[0:5]
    # generated = generated.unsqueeze(0)

    # res = model.generate(idx, max_new_tokens=100):
    last_token = 0
    with torch.no_grad():
        for _ in range(max_length):
            outputs, _, _ = model(generated)
            next_token_logits = outputs[0, -1, :] / temperature
            next_token_logits[last_token] = -10000
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, 1)
            
            if next_token != last_token:
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            
            last_token = next_token        
            # if next_token.item() == tokenizer.sep_token_id:
            #     break
    # print(generated[0])
    return tokenizer.decode(generated[0], skip_special_tokens=True)
    # return tokenizer.decode(list(generated[0]))

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits







dataset = MessageDataset_parquet(max_count=-1)



print("initializing model")
# very small 6M
# ntokens = dataset.num_words  # size of vocabulary
# emsize = 64  # embedding dimension
# d_hid = 128  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 4  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 4  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.2  # dropout probability

# small 16M (26m gpt2)
# ntokens = dataset.num_words  # size of vocabulary
# emsize = 256  # embedding dimension
# d_hid = 256  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 2  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.2  # dropout probability

# ntokens = dataset.num_words  # size of vocabulary
# emsize = 400  # embedding dimension
# d_hid = 400  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 2  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.2  # dropout probability

# 42M/57 parameter 
# ntokens = dataset.num_words  # size of vocabulary
# emsize = 400  # embedding dimension
# d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 12  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 8  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.2  # dropout probability

# 203M parameter
# ntokens = dataset.num_words  # size of vocabulary
# emsize = 1024  # embedding dimension
# d_hid = 2048  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 12  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 16  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.2  # dropout probability


# 264M/304M parameter
# ntokens = dataset.num_words  # size of vocabulary
# emsize = 1024  # embedding dimension
# d_hid = 4096  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 16  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 16  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.2  # dropout probability

# mobilellm shape(124.6M - really 151M)
# ntokens = dataset.num_words  # size of vocabulary
# emsize = 576  # embedding dimension
# d_hid = 1536  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 30  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 9  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.1  # dropout probability

#30m
# ntokens = dataset.num_words  # size of vocabulary
# emsize = 256  # embedding dimension
# d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 6  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 8  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.1  # dropout probability



ntokens = dataset.num_words  # size of vocabulary
emsize = 512  # embedding dimension
d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 6  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # number of heads in ``nn.MultiheadAttention``
dropout = 0.1  # dropout probability

block_size=max_seq_len
# model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, max_seq_len).to(device)
model = Transformer_Model(ntokens, block_size, emsize, nhead, nlayers, dropout, device=device).to(device)
 # 2048 1024 256 2 4 0.1

# save_name = "llm_304m_gpt2_512context.pt"
# save_name = "llm_57m_gpt2_weekend_mod.pt"
# save_name = "llm_mobilellm.pt"




# save_name = "llm_26m_gpt2.pt"
# save_name = "llm_57m_gpt2_weekend_mod.pt"
# save_name = "llm_42m2.pt"
# model.half()
# model = model.to(torch.bfloat16)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("num parameters", str(int(params/10**6)) + "M")


criterion = nn.CrossEntropyLoss()


# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)#, weight_decay=0.01)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)



# folders = os.listdir("models")




if True:#save_dir not in folders or "model.pt" not in os.listdir(os.path.join("models", save_dir)):
    print("making new")
    # if save_dir not in folders:
    #     os.mkdir(os.path.join("models", save_dir))

    train_info = {"dataset_ind": 0, "train_losses": [], "val_losses": [], "scheduler_count": 0}
    model.load_state_dict(torch.load("/home/nathan/Desktop/llm/model70m.pt", map_location=device))

else:
    print("loading")
    model.load_state_dict(torch.load(os.path.join("models", save_dir, "model.pt"), map_location=device))
    # 
    train_info = {"dataset_ind": 7, "train_losses": [], "val_losses": [], "scheduler_count": 0}
    # with open(os.path.join(os.path.join("models", save_dir, "train_info.json"))) as json_file:
    #     train_info = json.load(json_file)

if len(train_info["train_losses"]) >= 3:
       make_plot(train_info["train_losses"])







overused_words = []

last_save_time = time.time()
last_val_time = time.time()
gpu_time = 0
start_time = time.time()
model_save_num = 0



dataset.dataset_ind = train_info["dataset_ind"]-1


dataset.load_dataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)




optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# warmup_epochs = 100#(33e6*30) / tokens_per_batch * 0.1  # Number of epochs for the warm-up phase  (# tokens of dataset / tokens_per_batch * 0.1)
# max_epochs =  1000#(330e6*30) / tokens_per_batch # Total number of epochs (should match your training loop)
# scheduler = WarmupCosineLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs, warmup_start_lr=1e-8, eta_min=0)

steps_per_epoch = int(dataset.tokens_per_pq*len(dataset.datasets)/tokens_per_batch)
steps_per_epoch = 3000
print("steps per epoch", steps_per_epoch)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=steps_per_epoch)

if save_dir not in folders or "model.pt" not in os.listdir(os.path.join("models", save_dir)):
    pass
else:
    # pass
    optimizer.load_state_dict(torch.load(os.path.join("models", save_dir, "optimizer.pt")))
    # scheduler.load_state_dict(torch.load(os.path.join("models", save_dir, "scheduler.pth")))
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

tokens_processed = 0
last_sequence_token = 0
epoch_losses = []
step_losses = []
for epoch in range(80):
    
    msg_ind = 0
    many_loss = []
    batch_losses = []
    print("epoch", epoch)
    dataset.msg_size = -1
    dataset.batch_start = -1
    model.train()
    optimizer.zero_grad()

    print_loss_sum = 0
    print_loss_count = 0

    batch_accumulation_count = 0
    grad_accumulation_count = 0
    grad_loss = 0
    evaled_count = 0



    min_remainder = 24
    
    grad_accumulation_losses = []
    epoch_time = time.time()
    process_time = 0
    for message, s in dataloader:
        s = torch.max(s)

        process_t = time.time()
        message = message.to(device)

        message = message[:, 0:s] # truncate to min sequence length

        dataset.msg_size = -1
        dataset.batch_start = -1
        # print(message.shape)
        

        msg_ind += 1
        
        loss = 0

        src = message[:, 0:-1]

        target = message[:, 1::]
        # target.half()

        # src = src.T
        # target = target.T
        t = time.time()
        # idx torch.Size([64, 1024])
        # targ torch.Size([64, 1024])
        # print("max", torch.max(target), torch.max(src))
    
        try:
            with autocast():
                output, loss, tokens_used = model(src, targets=target)
        except:
            print("Forward error")
            optimizer.zero_grad()
            grad_loss = 0
            evaled_count = 0
            torch.cuda.empty_cache()
            continue

        grad_loss += loss.item() * tokens_used

        gpu_time += time.time() - t

        
        # ec = torch.count_nonzero(target).item()
        evaled_count += tokens_used

        batch_accumulation_count += 1
        t = time.time()
        
        gpu_time += time.time() - t

        
        min_remainder = min(min_remainder, (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9)

        t = time.time()

        
        try:
            scaler.scale(loss).backward()
        except: 
            print("likely memory allocation issue during backpropagation", (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/10**9, "GB left", message.shape, )
            optimizer.zero_grad()
            grad_loss = 0
            evaled_count = 0
            torch.cuda.empty_cache()


        gpu_time += time.time() - t


        if evaled_count > tokens_per_batch:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data /= evaled_count

            t=time.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()
            
            try:
                scaler.step(optimizer)
                scaler.update()
            except:
                print("error stepping optimizer")
            # scheduler.step()

            
            optimizer.zero_grad()
            gpu_time += time.time() - t
            # evaled_count = 1
            grad_accumulation_count = 0
            l = grad_loss/evaled_count
            grad_loss = 0

            batch_losses.append(l)
            print_loss_sum += l

            step_losses.append(l)

            train_info["scheduler_count"] += 1

            if train_info["scheduler_count"] % 20 == 0:

                sl = sum(step_losses)/len(step_losses)
                step_losses = []
                train_info["train_losses"].append(int(sl*10000)/10000)

                if len(train_info["train_losses"]) == 3:
                    make_plot(train_info["train_losses"])
                elif len(train_info["train_losses"]) > 3:
                    update_plot(train_info["train_losses"])



            # print("evaled count", evaled_count*1000)
            tokens_processed += evaled_count
            batch_tokens_processed = evaled_count
            evaled_count = 0
            print_loss_count += 1

        loss = 0

      
        if len(batch_losses) > 0 and print_loss_count >= 1:


            # print(dataset.tokenizer.decode(message[0, :], skip_special_tokens=True))
            eta = int((time.time()-epoch_time) / msg_ind * (len(dataloader)-msg_ind))
            if len(batch_losses) > 35:
                print("batch %d / %d loss: %4f recent loss: %4f epoch loss: %.4f epoch eta %d min_free %.1f lr %.7f" 
                    % (msg_ind, len(dataloader), print_loss_sum/print_loss_count, sum(batch_losses[-30::])/30, sum(batch_losses)/len(batch_losses), eta, min_remainder, optimizer.param_groups[0]['lr']))

            else:
                print("batch %d / %d loss: %4f epoch loss: %.4f epoch eta %d min_free %.1f lr %.7f"
                    % (msg_ind, len(dataloader), print_loss_sum/print_loss_count, sum(batch_losses)/len(batch_losses), eta, min_remainder, optimizer.param_groups[0]['lr']))



            # print(ts/gpu_time)
            print_loss_count = 0
            print_loss_sum = 0
            min_remainder = 24
            max_grad_accum = 0


        if time.time() - last_val_time > gen_text_time:
            print("time", time.time()-start_time, "gpu percent", gpu_time/(time.time()-start_time), "dataloading time", 1-process_time/(time.time()-epoch_time))

            print("Generated text:")

            prompts = ["The quick brown fox jumped", "I'm sorry but", "My favorite thing", "In the government", "I yelled at", "A popular game",
                      "I always wanted", "Dogs are nice", "There may be enough", "The oldest thing", "There are many books", "A computer can"]
            prompts = ["At the fruit store they sell tasty fruit like", "In the kitchen, I use many tools such as", 
                        "When it's cold outside, I like to wear warm clothes like a", "At the zoo, we saw many animals including",
                        "For breakfast, I usually have a healthy meal of", "In my garden, I grow different types of flowers such as",
                        "At the beach, you can enjoy activities like", "In school, students learn various subjects including",
                        "When cleaning the house, I use different tools like a", "At the playground, children enjoy playing on the",
                        "In the forest, you might see wildlife such as", "In my opinion, the most important invention is", "My favorite childhood memory is"]

            prompts = ["There are", "When a", "Unless a", "There will be", "There used to be", "A species of"]

            # prompts = ["Once upon a time there was a fish", "There used to be a"]
            for i in range(2):
                prompt = prompts[int(np.random.random()*len(prompts))]
                prompts.remove(prompt)
                generated_text = generate_text(model, dataset.tokenizer, prompt, max_length=30, temperature=0.6, top_p=0.9, device=device)

                print(generated_text)
            print("")

            last_val_time = time.time()
            model.train()

        if time.time() - last_save_time > checkpoint_save_freq or "stop" in os.listdir(os.path.join("models", save_dir)):
            print("Saving checkpoint")
            torch.save(model.state_dict(), os.path.join("models", save_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", save_dir, "optimizer.pt"))
            # torch.save(scheduler.state_dict(), os.path.join("models", save_dir, "scheduler.pth"))
            train_info_str = json.dumps(train_info)
            with open(os.path.join("models", save_dir, "train_info.json"), 'w') as fileHandle:
                fileHandle.write(str(train_info_str))
                fileHandle.close()
            print("saved checkpoint")
            last_save_time = time.time()
            if "stop" in os.listdir(os.path.join("models", save_dir)):
                exit()


        process_time += time.time() - process_t



    epoch_losses.append(sum(batch_losses) / len(batch_losses))
    print("epoch", epoch, "losses", epoch_losses)
    torch.cuda.empty_cache()


    train_info["dataset_ind"] += 1
    print("saving checkpoint")
    torch.save(model.state_dict(), os.path.join("models", save_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join("models", save_dir, "optimizer.pt"))
    # torch.save(scheduler.state_dict(), os.path.join("models", save_dir, "scheduler.pth"))
    train_info_str = json.dumps(train_info)
    with open(os.path.join("models", save_dir, "train_info.json"), 'w') as fileHandle:
        fileHandle.write(str(train_info_str))
        fileHandle.close()
    print("saved checkpoint")

    dataset.load_dataset()
    # torch.save(model.state_dict(), "llm_simple_conv_large.pt")

