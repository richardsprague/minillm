"""
Tokenizer management for MinillM.
Handles tokenizer loading and special token management.
"""

import os
from typing import List, Union, Optional
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

from .config import PathsConfig, TokensConfig


class TokenizerManager:
    """
    Manages tokenizer loading and special token handling.
    
    Provides a clean interface for tokenization with proper special token support.
    """
    
    def __init__(self, paths_config: PathsConfig, tokens_config: TokensConfig):
        self.paths_config = paths_config
        self.tokens_config = tokens_config
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self) -> None:
        """Load the ByteLevelBPE tokenizer from files."""
        try:
            self.tokenizer = ByteLevelBPETokenizer.from_file(
                vocab_filename=self.paths_config.vocab_path,
                merges_filename=self.paths_config.merges_path
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokens_config.pad_token
    
    @property
    def question_end_token_id(self) -> int:
        """Get question end token ID."""
        return self.tokens_config.question_end_token
    
    @property
    def answer_end_token_id(self) -> int:
        """Get answer end token ID."""
        return self.tokens_config.answer_end_token
    
    @property
    def think_start_token_id(self) -> int:
        """Get think start token ID."""
        return self.tokens_config.think_start_token
    
    @property
    def think_end_token_id(self) -> int:
        """Get think end token ID."""
        return self.tokens_config.think_end_token
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def decode(self, token_ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode (single sequence or batch)
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text (single string or list of strings)
        """
        if isinstance(token_ids[0], list):
            # Batch decoding
            return [
                self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
                for ids in token_ids
            ]
        else:
            # Single sequence decoding
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def encode_conversation(self, messages: List[dict]) -> List[int]:
        """
        Encode a conversation with proper special tokens.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Encoded conversation as token IDs
        """
        conversation_tokens = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'user':
                # Add question content + question end token
                tokens = self.encode(content, add_special_tokens=False)
                tokens.append(self.question_end_token_id)
                conversation_tokens.extend(tokens)
                
            elif role == 'assistant':
                # Handle thinking if present
                if 'thinking' in message:
                    thinking = message['thinking']
                    conversation_tokens.append(self.think_start_token_id)
                    conversation_tokens.extend(self.encode(thinking, add_special_tokens=False))
                    conversation_tokens.append(self.think_end_token_id)
                
                # Add assistant response + answer end token
                tokens = self.encode(content, add_special_tokens=False)
                tokens.append(self.answer_end_token_id)
                conversation_tokens.extend(tokens)
        
        return conversation_tokens
    
    def decode_conversation(self, token_ids: List[int]) -> List[dict]:
        """
        Decode token IDs back to conversation format.
        
        Args:
            token_ids: Encoded conversation tokens
            
        Returns:
            List of message dicts
        """
        messages = []
        current_tokens = []
        current_role = None
        in_thinking = False
        thinking_tokens = []
        
        for token_id in token_ids:
            if token_id == self.question_end_token_id:
                # End of user question
                if current_tokens:
                    content = self.decode(current_tokens, skip_special_tokens=True)
                    messages.append({'role': 'user', 'content': content})
                    current_tokens = []
                current_role = 'assistant'
                
            elif token_id == self.answer_end_token_id:
                # End of assistant answer
                if current_tokens:
                    content = self.decode(current_tokens, skip_special_tokens=True)
                    message = {'role': 'assistant', 'content': content}
                    
                    # Add thinking if we had any
                    if thinking_tokens:
                        thinking_content = self.decode(thinking_tokens, skip_special_tokens=True)
                        message['thinking'] = thinking_content
                        thinking_tokens = []
                    
                    messages.append(message)
                    current_tokens = []
                current_role = 'user'
                
            elif token_id == self.think_start_token_id:
                # Start of thinking
                in_thinking = True
                
            elif token_id == self.think_end_token_id:
                # End of thinking
                in_thinking = False
                
            else:
                # Regular token
                if in_thinking:
                    thinking_tokens.append(token_id)
                else:
                    current_tokens.append(token_id)
                    if current_role is None:
                        current_role = 'user'
        
        # Handle any remaining tokens
        if current_tokens:
            role = current_role if current_role else 'user'
            content = self.decode(current_tokens, skip_special_tokens=True)
            messages.append({'role': role, 'content': content})
        
        return messages
    
    def get_special_tokens_dict(self) -> dict:
        """Get dictionary of special tokens and their IDs."""
        return {
            'pad_token': self.pad_token_id,
            'question_end_token': self.question_end_token_id,
            'answer_end_token': self.answer_end_token_id,
            'think_start_token': self.think_start_token_id,
            'think_end_token': self.think_end_token_id,
        }
    
    def validate_tokenizer(self) -> bool:
        """
        Validate that the tokenizer is working correctly.
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Test basic encoding/decoding
            test_text = "Hello, world!"
            tokens = self.encode(test_text)
            decoded = self.decode(tokens)
            
            if not isinstance(tokens, list) or not tokens:
                return False
            
            if not isinstance(decoded, str):
                return False
            
            # Test special tokens
            special_tokens = self.get_special_tokens_dict()
            for token_name, token_id in special_tokens.items():
                if not isinstance(token_id, int) or token_id < 0:
                    print(f"Invalid special token {token_name}: {token_id}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Tokenizer validation failed: {e}")
            return False

    @classmethod
    def from_config(cls, paths_config: PathsConfig, tokens_config: TokensConfig) -> "TokenizerManager":
        """Create TokenizerManager from configuration."""
        return cls(paths_config, tokens_config)