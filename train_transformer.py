from torch.utils.data import Dataset, DataLoader
import torch
# import torch.nn as nn
import json
import numpy as np
import math
# from transformers import BertTokenizer
from transformers import GPT2Tokenizer


import pandas as pd

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import time
import os
# from torch.cuda.amp import autocast, GradScaler
# scaler = GradScaler()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

batch_size = 10
grad_accumulation_size = 86


from torch import nn, Tensor

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None, pad_mask: bool = False) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        if pad_mask:
            pad_mask = (src==0)
            # print(pad_mask)
            pad_mask = pad_mask.t()
        else:
            pad_mask = None
        # print(pad_mask.shape, src.shape)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(src.device)
        # src_mask = src_mask.to(torch.bfloat16)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=pad_mask)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def create_padding_mask(seq: Tensor, pad_token: int) -> Tensor:
    """
    Create a padding mask for the given sequence.
    Arguments:
        seq: Tensor, shape ``[batch_size, seq_len]``
        pad_token: int, the value used for padding

    Returns:
        mask: Tensor, shape ``[batch_size, seq_len]``, where positions of padding tokens are 1, else 0
    """
    return (seq == pad_token).transpose(0, 1)  # Transpose to shape [seq_len, batch_size]





class MessageDataset_parquet(Dataset):
    def __init__(self, max_count=-1, batch_size=16):

        self.batch_size = batch_size
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        

        self.pth = "fineweb_parquet_clean"
        ds = os.listdir(self.pth)
        self.datasets = []
        for d in ds:
            if d[-5::] == "rquet":
                self.datasets.append(d)
        self.datasets.sort()
        self.datasets.reverse()
        print(self.datasets)
        self.dataset_ind = 1
        vocabulary = self.tokenizer.get_vocab()
        self.num_words = len(vocabulary)

        self.load_dataset()


    def load_dataset(self):
        self.loaded = 0
        
        self.dataset_ind = (self.dataset_ind + 1) % len(self.datasets)

        
        n = self.datasets[self.dataset_ind]

        print("loading dataset", n)
        df = pd.read_parquet(os.path.join(self.pth, n), engine='pyarrow')

        print("loaded. sorting now")

        df['text_length'] = df['text'].apply(len)
        df_sorted = df.sort_values(by='text_length')
        df_sorted = df_sorted.drop(columns=['text_length'])
        
        lens = [len(a) for a in df["text"]]
        print("avg", sum(lens)/len(lens), max(lens), min(lens))

        df_sorted['text_length'] = df_sorted['text'].apply(len)
        df_filtered = df_sorted[(df_sorted['text_length'] >= 100) & (df_sorted['text_length'] <= 3000)]
        df_filtered = df_filtered.drop(columns=['text_length'])
        df_filtered = df_filtered.reset_index(drop=True)


        self.conversations = df_filtered.reset_index(drop=True)





        self.num_permutations = int(len(self.conversations) / self.batch_size)

        self.msg_size = -1
        self.batch_start = -1

        print("loaded dataset.", len(self.conversations), "conversations")



    def __len__(self):
        return self.num_permutations


    
    def __getitem__(self, idx):
        self.loaded += 1
        if self.loaded < 4:
            idx = self.num_permutations - self.loaded

        batch_start = (idx)*self.batch_size
    

        # print(tensor_list)

        tensor_list = [torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lst)), dtype=torch.long) for lst in self.conversations["text"][batch_start:batch_start+self.batch_size]]
        # for i, t in enumerate(tensor_list):
        #     if torch.isnan(t).any():
        #         print("has nan")
        #         print([lst for lst in self.conversations["text"][batch_start:batch_start+self.batch_size]])
        #         exit()

        # print(tensor_list)
        inputs = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0)
        # print(inputs.shape)

        # print(tensor_list.shape)

        return inputs.to(device)





dataset = MessageDataset_parquet(max_count=-1, batch_size=batch_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)



print("initializing model")
if False: # very small 6M
    ntokens = dataset.num_words  # size of vocabulary
    emsize = 64  # embedding dimension
    d_hid = 128  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 4  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 4  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
if True: # small 16M (26m gpt2)
    ntokens = dataset.num_words  # size of vocabulary
    emsize = 256  # embedding dimension
    d_hid = 256  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
if False: # small 26M
    ntokens = dataset.num_words  # size of vocabulary
    emsize = 400  # embedding dimension
    d_hid = 400  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
# elif True: #44M
#     ntokens = dataset.num_words  # size of vocabulary
#     emsize = 512  # embedding dimension
#     d_hid = 600  # dimension of the feedforward network model in ``nn.TransformerEncoder``
#     nlayers = 8  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
#     nhead = 8  # number of heads in ``nn.MultiheadAttention``
#     dropout = 0.2  # dropout probability
# else:
#     ntokens = dataset.num_words  # size of vocabulary
#     emsize = 512  # embedding dimension
#     d_hid = 800  # dimension of the feedforward network model in ``nn.TransformerEncoder``
#     nlayers = 12  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
#     nhead = 16  # number of heads in ``nn.MultiheadAttention``
#     dropout = 0.2  # dropout probability

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




model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

save_name = "llm_26m_gpt2.pt"
model.load_state_dict(torch.load(save_name, map_location=device))
save_name = "llm_26m_gpt2.pt"

# save_name = "llm_42m2.pt"
# model.half()
# model = model.to(torch.bfloat16)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("num parameters", str(int(params/10**6)) + "M")


criterion = nn.CrossEntropyLoss()


# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)#, weight_decay=0.01)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, device='cuda'):
    model.eval()
    generated = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            
            if next_token.item() == tokenizer.sep_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

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


overused_words = []

last_save_time = time.time()
last_val_time = time.time()
gpu_time = 0
start_time = time.time()
model_save_num = 0

ts = 0

epoch_losses = []
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
    max_batch_accum = 0

    grad_accumulation_losses = []

    for message in dataloader:
        message = message[0, :, :]

        dataset.msg_size = -1
        dataset.batch_start = -1
        # print(message.shape)
        

        msg_ind += 1
        
        loss = 0

        src = message[:, 0:-1]

        target = message[:, 1::]
        # target.half()

        src = src.T
        target = target.T

        t = time.time()
        
        try:
            output = model(src, pad_mask=True)
        except:
            print("forward error", src.shape, target.shape, grad_accumulation_count, torch.cuda.memory_allocated())
            optimizer.zero_grad()
            evaled_count = 0
            continue
        gpu_time += time.time() - t


        # output = output.float()
        # print(output.shape)
        output = output.reshape(-1, dataset.num_words)
        target = target.reshape(-1)

        try:
            if torch.isnan(output).any():
                print("output result nan")
                optimizer.zero_grad()
                evaled_count = 0
                continue

            if torch.isnan(target).any():
                print("target result nan")
                optimizer.zero_grad()
                evaled_count = 0
                continue
        except:
            print("error checking if nan")
            optimizer.zero_grad()
            evaled_count = 0
            continue
        ec = float(torch.sum(target!=0)/(10**3))
        evaled_count += ec
        # print(target.shape, target.dtype, output.dtype, output.dtype) # torch.Size([4, 16, 30522]) torch.float32 torch.Size([4, 16]) torch.int64

        t = time.time()
        try:
            loss = criterion(output[target!=0], target[target!=0]) * ec
            # print(criterion(output[target!=0], target[target!=0]))
            grad_loss += loss.item()
            
            batch_accumulation_count += 1
        except:
            print("error calculating loss. probably out of memory.")
            grad_loss = 0
            batch_accumulation_count = 0
            evaled_count = 0
        gpu_time += time.time() - t


        min_remainder = min(min_remainder, (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/10**9)

        t = time.time()
        try:
            loss.backward()
            grad_accumulation_count += 1
        except:
            print("likely memory allocation issue during backpropagation", (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/10**9, "GB left")
            optimizer.zero_grad()
            grad_loss = 0
            evaled_count = 0
            torch.cuda.empty_cache()
        gpu_time += time.time() - t
        ts += time.time()-t


        if grad_accumulation_count % grad_accumulation_size == 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data /= evaled_count

            t=time.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            gpu_time += time.time() - t
            
            grad_accumulation_count = 0
            l = grad_loss/evaled_count
            grad_loss = 0

            batch_losses.append(l)
            print_loss_sum += l

            evaled_count = 0
            print_loss_count += 1

        loss = 0

      
        if len(batch_losses) > 0 and print_loss_count >= 1:
            print("batch", msg_ind, "/", len(dataloader), "loss", print_loss_sum/print_loss_count, "epoch loss", sum(batch_losses)/len(batch_losses), "min free", min_remainder)
            # print(ts/gpu_time)
            print_loss_count = 0
            print_loss_sum = 0
            min_remainder = 24
            max_grad_accum = 0


        if time.time() - last_val_time > 800:
            print("time", time.time()-start_time, "gpu percent", gpu_time/(time.time()-start_time))

            print("Generated text:")

            prompts = ["The quick brown fox jumped", "I'm sorry but", "My favorite thing", "In the government", "I yelled at", "A popular game",
                      "I always wanted", "Dogs are nice", "There may be enough", "The oldest thing", "There are many books", "A computer can"]
            for i in range(2):
                prompt = prompts[int(np.random.random()*len(prompts))]
                prompts.remove(prompt)
                generated_text = generate_text(model, dataset.tokenizer, prompt, max_length=30, temperature=0.7, top_p=0.9, device=device)
            
                print(generated_text)
            print("")

            last_val_time = time.time()
            model.train()
    
        if time.time() - last_save_time > 500: #* (model_save_num+1)):
            model_save_num += 1
            print("saving model - time", model_save_num)


            torch.save(model.state_dict(), save_name)
            last_save_time = time.time()
            # if model_save_num == 50:
            #     exit()
    epoch_losses.append(sum(batch_losses) / len(batch_losses))
    print("epoch", epoch, "losses", epoch_losses)
    torch.cuda.empty_cache()

    dataset.load_dataset()
    # torch.save(model.state_dict(), "llm_simple_conv_large.pt")

