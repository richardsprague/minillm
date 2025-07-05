"""
Text generation utilities for MinillM.
Handles efficient text generation with advanced sampling techniques.
"""

from typing import List, Dict, Optional, Any
import torch
import torch.nn.functional as F

from .config import GenerationConfig
from .tokenizer import TokenizerManager
from .models import TransformerModel


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
        model: TransformerModel, 
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
        
        # Encode conversation
        conversation_tokens = self.tokenizer.encode_conversation(conversation)
        
        # Convert to tensor
        tokens = torch.tensor([conversation_tokens], dtype=torch.long, device=next(self.model.parameters()).device)
        
        # Clear model cache for new generation
        self.model.clear_kv_cache()
        
        # Prefill the model with conversation context
        with torch.no_grad():
            if tokens.shape[1] > 1:
                # Process all but the last token for KV cache
                _ = self.model(tokens[:, :-1], start_pos=0)
        
        # Generate response tokens
        generated_tokens = []
        current_pos = tokens.shape[1] - 1
        
        # Track recent tokens for repetition penalty
        recent_tokens = list(conversation_tokens[-50:]) if len(conversation_tokens) > 50 else list(conversation_tokens)
        
        with torch.no_grad():
            for step in range(max_length):
                # Get logits for next token
                logits = self.model(tokens[:, -1:], start_pos=current_pos)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0 and recent_tokens:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, recent_tokens, repetition_penalty
                    )
                
                # Apply top-k filtering
                if top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                # Sample next token
                if self.config.do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                next_token_id = next_token.item()
                
                # Check for end tokens
                if next_token_id in [
                    self.tokenizer.answer_end_token_id,
                    self.tokenizer.question_end_token_id
                ]:
                    break
                
                # Handle newline repetition bug
                newline_tokens = [208, 230, 15078, 19]  # Common newline token IDs
                if (len(generated_tokens) > 0 and 
                    generated_tokens[-1] in newline_tokens and 
                    next_token_id in newline_tokens):
                    # Skip repeated newlines
                    continue
                
                # Add token to generation
                generated_tokens.append(next_token_id)
                recent_tokens.append(next_token_id)
                
                # Keep recent_tokens list manageable
                if len(recent_tokens) > 100:
                    recent_tokens = recent_tokens[-50:]
                
                # Update tokens tensor for next iteration
                # Ensure next_token has the right shape [batch_size, 1]
                next_token = next_token.view(tokens.shape[0], 1)
                tokens = torch.cat([tokens, next_token], dim=-1)
                current_pos += 1
                
                # Stream output if requested
                if stream:
                    token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                    print(token_text, end='', flush=True)
        
        # Decode generated tokens to text
        if generated_tokens:
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
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