from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import json
import numpy as np
import math
import random

import csv
# from transformers import GPT2Tokenizer
from tokenizers import ByteLevelBPETokenizer


import re

from transformers import get_cosine_schedule_with_warmup

import pandas as pd

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import time
import os
from torch.cuda.amp import autocast, GradScaler
# from transformer_model_llama_june2025_checkpointing import TransformerModel
from transformer_model_llama_june2025 import TransformerModel

scaler = GradScaler()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("using", device)

using_gpu = torch.cuda.is_available()
if using_gpu:
    torch.cuda.empty_cache()


save_path = "models"


gen_text_time = 300
max_seq_len_given = 4096#1024

# save_dir = "138m_gpt2_high_lr2"
save_dir = "llama_505m_my_tokenizer2"
# save_dir = "llama_super_small_my_tokenizer"

batch_size = 1#6#12
effective_batch_size = 128#256#1024#2048#256#512

checkpoint_save_freq = 500000
lr = 1e-5
full_dataset_size = 30e9 # 40b




tokens_per_batch = (max_seq_len_given) * effective_batch_size
print("tokens per batch", tokens_per_batch)

tokens_per_plot = 5e7
use_scheduler = True


ntokens = 50257 # gpt2
# ntokens = 24000 # mytokenizer_sep27
ntokens = 50000 # 50k

max_seq_len = max_seq_len_given+1

# max_seq_len = 1024
# block_size=max_seq_len


emsize = -1  # embedding dimension
d_hid = 6144  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 24  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 16  # number of heads in ``nn.MultiheadAttention``
dropout = 0.01  # dropout probability
ntokens = 50000
block_size = 1024
ffn_dim = 4096
dim = 1024




tokenizer = ByteLevelBPETokenizer.from_file(
        vocab_filename="my_tokenizer_50k_2025/tokenizer_50k_2025-vocab.json",
        merges_filename="my_tokenizer_50k_2025/tokenizer_50k_2025-merges.txt"
    )

print(tokenizer.decode([0,1,2,3,4,5,6,7,8]))

question_end_token=1
answer_end_token=2
think_start_token = 7
think_end_token = 8
pad_token_id=0

model = TransformerModel(ntokens, max_seq_len, emsize, nhead, nlayers, ffn_dim=ffn_dim, dim=dim, batch_size=batch_size, device=device).to(device)


print("checking model function")
model(torch.tensor([[1,2,3]], dtype=torch.long).to(device), start_pos=0)




# model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("num parameters", str(int(params/10**6)) + "M")


class MessageDataset(Dataset):
    def __init__(self):

        self.jsonl = True


        self.tokenizer = tokenizer

        self.pth = "/media/nathan/EB37-4BA6/llm_big/smollm/csv/google_natural_questions"
        self.pth = "/media/nathan/EB37-4BA6/llm_big/smollm/csv/ultrachat"
        self.pth = "/media/nathan/EB37-4BA6/llm_big/raw_datasets/smol_smoltalk"
        self.pth = "/media/evanslab/EB37-4BA6/llm_big/generate_datasets/regular_qa_to_reasoning"

        self.all_datasets = []
        files = os.listdir(self.pth)
        print(files)
        for f in files:
            if self.jsonl and f[-6::] == ".jsonl":
                # if "smoltalk" in f:
                self.all_datasets.append(os.path.join(self.pth, f))
            elif not self.jsonl and f[-4::] == ".csv":
                self.all_datasets.append(os.path.join(self.pth, f))


        self.chunk_size = 300
        
        self.dataset_ind = 0
        self.dataset_seq_ind = 0
        self.partial_loading = False


        self.tokenized_seq_ind = 0

        self.conversations = []

        self.contexts = {}

        self.locations = []
        
        vocabulary = self.tokenizer.get_vocab()

        self.num_words = len(vocabulary)
        print("vocab size", self.num_words)
        if abs(self.num_words-151936) < 4: # qwen is weird
            self.num_words = 151936

        if self.jsonl:
            self.load_dataset_jsonl()
        else:
            self.load_dataset_csv()
        # self.load_file_searcher()

    def remove_emojis(self, text):
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002700-\U000027BF"  # Dingbats
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002600-\U000026FF"  # Miscellaneous Symbols
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)


    def load_dataset_jsonl(self):

        self.conversations = []
        rows = []
        for i in range(len(self.all_datasets)):
            print("loading", self.all_datasets[self.dataset_ind])
            with open(self.all_datasets[self.dataset_ind], 'r') as file:
              for line in file:
                  try:
                      rows.append(json.loads(line)["data"])
                  except:
                      continue
            self.dataset_ind += 1
            if self.partial_loading:
                break

        # print(rows[10])
        # print(contexts)
        print("loaded.", len(rows), "rows. tokenizing")
        random.shuffle(rows)
        # for i in rows:
        #     print(i)
        # for i in rows:
        #     print(i)

        reasoning = True



        independent_tokenized = []
        rows_tokenized = []

        convs_truncated = 0
        incomplete_convs = 0
        convs_removed = 0
        tokens_used = 0
        for ind, r in enumerate(rows):
            # carryover = []
            # carryover_masks = []
            # print(r)
            
            all_conv_messages = []
            all_conv_masks = []
            messages = []
            masks = []
            #     break
            
            non_reasoning_tok = []
            non_reasoning_conv = []
            # print(r)

            num_exchanges = 0
            for i, c in enumerate(r):

                c = c.replace("\n###", "\n")
                c = c.replace("\n##", "\n")
                c = c.replace("**", "")
                c = c.replace("\n---\n", "")

                c = self.remove_emojis(c)

                if i%2==1:
                    while c[-1] not in ".!?":
                        c = c[0:-1]
                    while c[0] == "\n":
                        c = c[1::]
                

                if i%2 == 1 and "<think>" in c:
                    c = c.replace("<think>", "")
                    cc = c.split("</think>")
                    reasoning = True

                    # print(c)
                    # print(cc)
                    if len(cc) != 2:
                        convs_removed += 1
                        masks = []
                        messages = []
                        break
                    tok = tokenizer.encode(cc[0]).ids + [think_end_token] + tokenizer.encode(cc[1]).ids
                    non_reasoning_tok = tokenizer.encode(cc[1]).ids
                else:
                    tok = tokenizer.encode(c).ids

                if i%2 == 0: # question
                    if reasoning and i > 1:
                        all_conv_messages.append(messages)
                        all_conv_masks.append(masks)

                        messages = non_reasoning_conv[:]
                        masks = [0]*len(messages)
                        # break

                    masks += [0]*(len(tok)+1) # always masks
                    messages += tok + [question_end_token]
                    if reasoning:
                        messages.append(think_start_token)
                        masks.append(0)
                        non_reasoning_conv += tok + [question_end_token]
                else:
                    masks += [1]*(len(tok)+1)
                    messages += tok + [answer_end_token]
                    num_exchanges += 1
                    if reasoning:
                        non_reasoning_conv += non_reasoning_tok + [answer_end_token]
                
                if len(masks) >= max_seq_len:
                    break
            all_conv_messages.append(messages)
            all_conv_masks.append(masks)

            for masks, messages in zip(all_conv_masks, all_conv_messages):
                if sum(masks) > 0:
                    # print(len(masks))

                    if len(messages) <= max_seq_len:
                        self.conversations.append([messages, masks])
                        tokens_used += sum(masks)
                        # print("no trunc")

                    else:
                        self.conversations.append([messages[0:max_seq_len-1], masks[0:max_seq_len-1]])
                        tokens_used += sum(masks[0:max_seq_len-1])
                        # print("trunc")
                        convs_truncated += 1
                        if num_exchanges <= 1:
                            incomplete_convs += 1
            if ind % 5000 == 4999:
                print(ind, "/", len(rows))
                # break


            # if ind > 0:
            #     print("\n\nLIMITING CONVERSATION count\n\n")
            #     break
        # for inputs, masks in self.conversations:
        #     inputs = np.array(inputs)
        #     masks = np.array(masks)
        #    # print(tokenizer.decode(list(inputs)), "<-all\n")
        #     print(tokenizer.decode(list(inputs[masks==0])), "<- given")
        #     print(tokenizer.decode(list(inputs[masks==1])), "<- train\n")
        #     print("\n\n\n\n")
                

        # print("conversations", self.conversations)
        print("num convs", len(self.conversations), "|   truncated", convs_truncated, " | incomplete", incomplete_convs, " |  didn't use for length", convs_removed, "  |  tokens used", tokens_used)
        return tokens_used



    def load_file_csv(self, n):

        tokens_used = 0
        convs_removed = 0

        print("loading", n, self.dataset_ind)
        contexts = []
        rows = []
        with open(os.path.join(n), mode='r') as file:
            csv_reader = csv.reader(file)
            # Iterate over the rows
            for i, row in enumerate(csv_reader):
                if i == 0:
                    pass
                else:
                    if len(row) > 0:
                        rows.append(row)

                # if len(rows) > 200:
                #     break
        # print(rows[10])
        # print(contexts)
        print("loaded.", len(rows), "rows. tokenizing")
        random.shuffle(rows)
        # for i in rows:
        #     print(i)
        # for i in rows:
        #     print(i)

        independent_tokenized = []
        rows_tokenized = []



        for ind, r in enumerate(rows):
            # carryover = []
            # carryover_masks = []
            # print(r)
            
            messages = []
            masks = []
            #     break

            for i, c in enumerate(r):
                tok = tokenizer.encode(c).ids

                if i%2 == 0: # question
                    masks += [0]*(len(tok)+1) # always masks
                    messages += tok + [question_end_token]
                else:
                    masks += [1]*(len(tok)+1)
                    messages += tok + [answer_end_token]
                    tokens_used += len(tok)+1
                
                if len(masks) >= max_seq_len:
                    break

            if len(masks) > 0 and sum(masks) > 0:

                if len(messages) <= max_seq_len:
                    self.conversations.append([messages, masks])
                else:
                    self.conversations.append([messages[0:max_seq_len-1], masks[0:max_seq_len-1]])

            if ind % 5000 == 4999:
                print(ind, "/", len(rows))


            # if ind > 1000:
            #     print("LIMITING CONVERSATION LENGTH")
            #     break
        
        

        # print("conversations", self.conversations)
        print("num convs", len(self.conversations), "didn't use for length", convs_removed, "tokens used", tokens_used)
        return tokens_used


    def load_dataset_csv(self):
        self.conversations = []
        self.loaded = 0
        tokens_used = 0

        if self.partial_loading:
            print("partial loading")
            tokens_used += self.load_file_csv(self.all_datasets[self.dataset_ind])
            self.dataset_ind = (self.dataset_ind+1)%len(self.all_datasets)
        else:
            for n in self.all_datasets:
                tokens_used += self.load_file_csv(n)
                
        print("total tokens", tokens_used)




    def __len__(self):
        return len(self.conversations)


    
    def __getitem__(self, idx):
        s = len(self.conversations[idx][0])
        # print(len(self.conversations[idx][0]), len(self.conversations[idx][1]))
        if len(self.conversations[idx][0])<max_seq_len_given:
            self.conversations[idx][0] += [0]*(max_seq_len_given-len(self.conversations[idx][0]))
            self.conversations[idx][1] += [0]*(max_seq_len_given-len(self.conversations[idx][1]))

        inputs = torch.tensor(self.conversations[idx][0])
        masks = torch.tensor(self.conversations[idx][1])
        # s = self.conversations[idx][2]
        

        # print(masks)
        # a = tokenizer.decode(inputs[0:s], skip_special_tokens=False)
        # print(tokenizer.decode(list(inputs[0:s])), "<-all\n")
        # print(tokenizer.decode(list(inputs[0:s][masks[0:s]==0])), "<- given")
        # print(tokenizer.decode(list(inputs[masks==1])), "<- train\n")
        
        # print(list(inputs[masks==1]))
        # print('\n')

        # print("lengths", len(inputs), len(masks))

        
        return inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True)




# def generate_text(model, tokenizer, prompt, max_length=1, temperature=0.8, top_p=1, device='cuda'):
#     return ""

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, device='cuda'):
    model.eval()
    model.clear_kv_cache()
    tokens = torch.tensor([tokenizer.encode(prompt).ids + [question_end_token, 7]])

    generated = tokens.to(device)
    for g in range(generated.shape[1]):
        model(generated[:, g:g+1], start_pos=g)
    # res = model.generate(idx, max_new_tokens=100):
    last_token = 0
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated[:, -1::], start_pos=generated.shape[1])
            next_token_logits = outputs[0, :] / temperature
            next_token_logits[0, last_token] = -10000
            next_token_logits = next_token_logits.squeeze()
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)

            probabilities = F.softmax(filtered_logits, dim=-1)
            
            next_token = torch.multinomial(probabilities, 1)
            
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            
            if next_token == answer_end_token:
                break
            
            last_token = next_token

    # print(list(generated[0]))
    return tokenizer.decode(list(generated[0]))


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




dataset = MessageDataset()







criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.1)

if False:
    print("LOADING FINETUNED MODEL CHECKPOINT!")
    model.load_state_dict(torch.load("model_finetuned.pt", weights_only=True, map_location=device))
    optimizer.load_state_dict(torch.load("optimizer.pt", weights_only=True, map_location=device))
else:
    print("loading model weights")
    model.load_state_dict(torch.load("model.pt", weights_only=True, map_location=device))


print('testing text generation:')
generated_text = generate_text(model, dataset.tokenizer, "Who was the first president of the United States?", max_length=30, temperature=0.7, top_p=0.9, device=device)
print(generated_text)


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)






tokens_processed = 0
tokens_per_print = 0


model.train()
optimizer.zero_grad()

print_losses = []
losses_between_prints = []
batch_losses = []
step_losses = []

epoch_time = time.time()
last_gen_time = time.time()
last_print_time = time.time()
start_time = time.time()
total_steps = 0
current_steps = 0

torch.compile(model)

grad_accum_steps = int(effective_batch_size/batch_size)
accum_step = 0

for epoch in range(10):
    for message, masks in dataloader:

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(tokens=message[:, 0:-1], start_pos=0)
            

            output = output.reshape(-1, dataset.num_words)
            message = message[:, 1::].reshape(-1)
            masks = masks[:, 1::].reshape(-1) 
            loss_per_token = F.cross_entropy(output, message, reduction='none')  # (B*S,)
            masked_loss = loss_per_token * masks  # (B*S,)
            tokens_used = masks.sum().clamp(min=1)

            loss = masked_loss.sum()/tokens_used

            # print(loss)

            scaler.scale(loss / grad_accum_steps).backward()

        # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # loss = model(message[:, 0:-1], labels=message[:, 1::]).loss
        # tokens_used = message.shape[0]*message.shape[1]
        # scaler.scale(loss / grad_accum_steps).backward()
        # (loss/grad_accum_steps).backward()

        tokens_processed += tokens_used
        tokens_per_print += tokens_used
        accum_step +=1

        batch_losses.append(loss.item())
           
        if accum_step >= grad_accum_steps:

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            # print("stepped")

            if use_scheduler:
                correct_step_amt =  tokens_processed / full_dataset_size * total_steps
                while current_steps < correct_step_amt:
                    current_steps += 1
                    scheduler.step()
                    # print("step")

            optimizer.zero_grad()


            losses_between_prints.append(sum(batch_losses)/len(batch_losses))
            step_losses.append(sum(batch_losses)/len(batch_losses))

            batch_losses=[]
            accum_step = 0



        # print(tokens_per_print/1000000)
        if tokens_per_print >= 1e4 and len(losses_between_prints) > 0:

            tokens_per_second = tokens_per_print / (time.time()-last_print_time) / 1000
            tokens_per_print = 0

            print_losses.append(sum(losses_between_prints)/len(losses_between_prints))
            losses_between_prints = []


            if tokens_processed > 1e9:
                tp_suffix = "B"
                tp = tokens_processed / 1e9
            elif tokens_processed > 1e6:
                tp_suffix = "M"
                tp = tokens_processed / 1e6
            else:
                tp_suffix = "K"
                tp = tokens_processed / 1e3


            if len(print_losses) < 35:
                print("loss: %4f | tok speed %.1fk | lr %.7f | tp: %.3f%s | Runtime: %d"
                    % (sum(print_losses)/len(print_losses), tokens_per_second, optimizer.param_groups[0]['lr'], tp, tp_suffix, int(time.time()-start_time)))

            else:
                print("loss: %4f | recent loss: %4f | tok speed %.1fk | lr %.7f | tp: %.3f%s | Runtime: %d" 
                    % (sum(print_losses)/len(print_losses), sum(print_losses[-30::])/30, tokens_per_second, optimizer.param_groups[0]['lr'], tp, tp_suffix, int(time.time()-start_time)))

            if len(print_losses) > 35:
                print_losses = print_losses[-35::]

            last_print_time = time.time()



        if time.time() - last_gen_time > gen_text_time:

            print("Generated text:")

            prompts = ["Who was the first president of the United States?", "What is the capital of France?", "What can run faster, a turtle or a rabbit?", "What is the diameter of the earth"]
            for i in range(2):
                prompt = prompts[int(np.random.random()*len(prompts))]
                prompts.remove(prompt)
                generated_text = generate_text(model, dataset.tokenizer, prompt, max_length=80, temperature=0.7, top_p=0.9, device=device)

                print(generated_text)
            print("")

            last_gen_time = time.time()
            model.train()

    print("saving")
    torch.save(model.state_dict(), "model_finetuned.pt")
    torch.save(optimizer.state_dict(), "optimizer.pt")
    if epoch == 1:
        print("done")
        exit()
    print("new epoch")
    if dataset.partial_loading:
        if dataset.jsonl:
            dataset.load_dataset_jsonl()
        else:
            dataset.load_dataset_csv()