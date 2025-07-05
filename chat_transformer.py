from transformer_model_llama_june2025 import TransformerModel
import torch
import torch.nn.functional as F
import os
from tokenizers import ByteLevelBPETokenizer



model_name = "../model505m_july3_2025/model505m_july3_2025.pt"
tokenizer_name = "../model505m_july3_2025/my_tokenizer_50k_2025"

torch.manual_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(model, tokenizer, tokens, max_length=100, temperature=0.7, top_p=0.9,
                  repetition_penalty=1.1, stream=False):
    model.clear_kv_cache()
    

    # print("prompt->", prompt)

    generated = tokens
    # for g in range(generated.shape[1]):
    #     model(generated[:, g:g+1], start_pos=g)
    model(generated[:, 0:-1], start_pos=0)
    thinking = True
    last_token = 0
    last_recent_tokens = []
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated[:, -1:], start_pos=generated.shape[1])
            next_token_logits = outputs[0, :] / temperature

            # Apply repetition penalty
            for token_id in set(last_recent_tokens):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty

            # # (Optional) prevent repeating last token explicitly
            # next_token_logits[0, last_token] -= 10

            # print("gen", generated[:, -1])
            newline_tokens = [208, 230, 15078, 19]#, 35, 23, 1820]
            if generated[:, -1].item() in newline_tokens and generated[:, -2].item() in newline_tokens: # bug where it generates new line forever
                for n in newline_tokens:
                    next_token_logits[0, n] -= 1000
                # print("PREVENTING TOKEN")

            # print("Gen->",generated[:, -1].item(), generated[:, -2].item())
            if generated[:, -1].item() == think_end_token:
                if thinking:
                    thinking = False
            if not thinking:
                next_token_logits[0, think_end_token] -= 1000

            next_token_logits = next_token_logits.squeeze()
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, 1)
            # if next_token != last_token:
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            last_recent_tokens.append(next_token.item())
            if len(last_recent_tokens)>100:
                last_recent_tokens = last_recent_tokens[1::]
            # print("next", next_token)
            last_token = next_token
            if last_token == answer_end_token:
                break

            if stream:
                print(tokenizer.decode([next_token.item()]), end='', flush=True)


    return generated[0].tolist()


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits





emsize = -1  # embedding dimension
d_hid = 6144  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 24  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 16  # number of heads in ``nn.MultiheadAttention``
dropout = 0.01  # dropout probability
ntokens = 50000
block_size = 4096
ffn_dim = 4096
dim = 1024

tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename=os.path.join(tokenizer_name, "tokenizer_50k_2025-vocab.json"),
    merges_filename=os.path.join(tokenizer_name, "tokenizer_50k_2025-merges.txt")
)

question_end_token = 1
answer_end_token = 2
think_start_token = 7
think_end_token = 8

# print(tokenizer.decode([0,1,2,3,4,5,6,7,8]))


model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, ffn_dim=ffn_dim, dim=dim, batch_size=1, device=device).to(device)
model.eval()


model.load_state_dict(torch.load(model_name, weights_only=True, map_location=device))


conversation = []
full_text = ""
while True:
    prompt = input("\n>>")

    if len(prompt) == 0:
       
        model.clear_kv_cache()
        print("reset")
        full_text = ""
        conversation = []
        continue

    conversation.append(tokenizer.encode(prompt).ids) 

    # print("\nconversation tokens", conversation)
    tokens = []
    for i, msg in enumerate(conversation):
        if i%2==0: # question
            tokens += msg + [question_end_token]
        else: # answer
            tokens += msg + [answer_end_token]

    pre_conv_len = len(tokens)-1
    tokens.append(think_start_token)

    # print("\ngiving->", tokenizer.decode(tokens), "\n")

    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    all_tokens = generate_text(model, tokenizer, tokens, stream=True, temperature=0.3, top_p=0.9, max_length=4096, repetition_penalty=1.2) 

    if all_tokens[-1] == answer_end_token:
        all_tokens = all_tokens[0:-1]

    if think_end_token in all_tokens:
        msg = all_tokens[all_tokens.index(think_end_token)+1::]
        conversation.append(msg)
    else:
        msg = all_tokens[pre_conv_len::]
        conversation.append(msg)


