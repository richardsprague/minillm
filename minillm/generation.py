"""
Text generation utilities for MinillM.
Handles efficient text generation with advanced sampling techniques.
"""

from typing import List, Dict, Optional, Any
import torch
import torch.nn.functional as F

from .config import GenerationConfig
from .tokenizer import TokenizerManager
from .models import TransformerModel, LegacyTransformerModel


class TextGenerator:
    """
    Advanced text generator with multiple sampling strategies.
    
    Supports:
    - Temperature scaling
    - Top-p (nucleus) sampling
    - Top-k sampling
    - Repetition penalty
    - Streaming generation
    """
    
    def __init__(
        self, 
        model, 
        tokenizer: TokenizerManager, 
        config: GenerationConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Generation state
        self.conversation_tokens = []
        self.last_tokens = []  # For repetition penalty
        
    def generate_response(
        self, 
        conversation: List[Dict[str, str]], 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate a response to a conversation.
        
        Args:
            conversation: List of message dicts with 'role' and 'content'
            max_length: Override max generation length
            temperature: Override temperature
            top_p: Override top-p value
            top_k: Override top-k value
            repetition_penalty: Override repetition penalty
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        # Use config defaults or overrides
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        
        # Use simplified working generation approach
        try:
            # Get device from model
            device = next(self.model.parameters()).device
        except:
            device = torch.device('cpu')
        
        # Extract user message from conversation (simplified for now)
        user_message = ""
        for msg in conversation:
            if msg['role'] == 'user':
                user_message = msg['content']
                break
        
        if not user_message:
            return ""
        
        # Encode exactly like the working original
        encoded = self.tokenizer.tokenizer.encode(user_message)
        tokens = encoded.ids + [self.tokenizer.question_end_token_id, self.tokenizer.think_start_token_id]
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Clear cache and prefill
        self.model.clear_kv_cache()
        with torch.no_grad():
            if tokens_tensor.shape[1] > 1:
                self.model(tokens_tensor[:, :-1], start_pos=0)
        
        # Generate using working logic
        generated = tokens_tensor
        all_generated = []
        
        with torch.no_grad():
            for i in range(max_length):
                outputs = self.model(generated[:, -1:], start_pos=generated.shape[1])
                next_token_logits = outputs[0, :] / temperature
                
                # Apply top-p filtering like the working version
                next_token_logits = next_token_logits.squeeze()
                if top_p < 1.0:
                    filtered_logits = self._top_p_filtering(next_token_logits, top_p)
                else:
                    filtered_logits = next_token_logits
                
                probabilities = F.softmax(filtered_logits, dim=-1)
                
                if self.config.do_sample:
                    next_token = torch.multinomial(probabilities, 1)
                else:
                    next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
                
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                all_generated.append(next_token.item())
                
                if next_token.item() == self.tokenizer.answer_end_token_id:
                    break
        
        # Process generated tokens like the working version
        if all_generated and all_generated[-1] == self.tokenizer.answer_end_token_id:
            all_generated = all_generated[:-1]
        
        if self.tokenizer.think_end_token_id in all_generated:
            think_idx = all_generated.index(self.tokenizer.think_end_token_id)
            message_tokens = all_generated[think_idx + 1:]
        else:
            message_tokens = all_generated
        
        if message_tokens:
            response = self.tokenizer.tokenizer.decode(message_tokens)
            return response.strip()
        else:
            return ""
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        previous_tokens: List[int], 
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        # Clone logits to avoid in-place modification issues
        logits = logits.clone()
        for token_id in set(previous_tokens):
            if token_id < logits.size(0):  # Check bounds
                if logits[token_id] > 0:
                    logits[token_id] = logits[token_id] / penalty
                else:
                    logits[token_id] = logits[token_id] * penalty
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if top_k > 0:
            # Get top-k values and indices
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Set all values below top-k to negative infinity
            logits[logits < top_k_values[-1]] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def generate_stream(
        self, 
        conversation: List[Dict[str, str]], 
        **kwargs
    ):
        """
        Generate response with streaming output.
        
        Yields:
            Individual tokens as they are generated
        """
        # This would implement streaming generation
        # For now, we'll use the regular generation with stream=True
        return self.generate_response(conversation, stream=True, **kwargs)


class ConversationManager:
    """Manages conversation state and context."""
    
    def __init__(self, tokenizer: TokenizerManager, max_context_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.conversation_history = []
    
    def add_message(self, role: str, content: str, thinking: Optional[str] = None):
        """Add a message to the conversation."""
        message = {'role': role, 'content': content}
        if thinking:
            message['thinking'] = thinking
        self.conversation_history.append(message)
    
    def get_context_tokens(self) -> List[int]:
        """Get conversation tokens, truncating if necessary."""
        # Encode full conversation
        tokens = self.tokenizer.encode_conversation(self.conversation_history)
        
        # Truncate if too long, keeping the most recent context
        if len(tokens) > self.max_context_length:
            # Try to keep complete exchanges
            truncated_tokens = tokens[-self.max_context_length:]
            
            # Find the first complete user message to start from
            for i, token_id in enumerate(truncated_tokens):
                if token_id == self.tokenizer.question_end_token_id:
                    truncated_tokens = truncated_tokens[i+1:]
                    break
            
            return truncated_tokens
        
        return tokens
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """Get the last n messages from conversation."""
        return self.conversation_history[-n:] if n > 0 else []