# Gemini LLM Project Guide

This guide provides step-by-step instructions to set up your environment and run the chat transformer.

## 1. Environment Setup

It is highly recommended to use a virtual environment to manage project dependencies.

### Create a Virtual Environment

Open your terminal in the project root directory and run the following command to create a virtual environment named `.venv`:

```bash
python3 -m venv .venv
```

### Activate the Virtual Environment

**On macOS and Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

You should see `(.venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.

### Install Dependencies

With the virtual environment active, install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
> **Note:** You can also use the VS Code task "Install dependencies" to run this command.

## 2. Download Model and Tokenizer

As mentioned in the `README.md`, you need to download the pretrained model and the tokenizer files.

1.  **Model:** Download the model from this Google Drive link and place it in the project directory. The expected filename is `model505m_july3_2025.pt`.
2.  **Tokenizer:** The code expects a tokenizer directory named `my_tokenizer_50k_2025`. Make sure you have this directory with the `vocab.json` and `merges.txt` files inside.

## 3. Configure Paths

Open `chat_transformer.py` and ensure the `model_name` and `tokenizer_name` variables at the top of the file point to your downloaded files.

```python
model_name = "model505m_july3_2025.pt"
tokenizer_name = "my_tokenizer_50k_2025"
```

## 4. Run the Chatbot

You are now ready to run the chatbot. Execute the following command in your terminal:

```bash
python chat_transformer.py
```

Alternatively, you can use the "Run Chat Transformer" task in VS Code.

After running, you can type your prompts in the terminal. To start a new conversation, just press Enter on an empty prompt.

## 5. Finetuning (Optional)

If you wish to finetune the model on your own dataset:

1.  Open `finetune_llama.py`.
2.  In the `MessageDataset` class, update the `self.pth` variable to the path of your dataset.
3.  Run the finetuning script:
    ```bash
    python finetune_llama.py
    ```
    You can also use the "Finetune Model" task in VS Code.