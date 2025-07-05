"""
Web server for MinillM using FastAPI.
Provides REST API endpoints for chat and generation.
"""

import asyncio
import time
from typing import List, Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .config import Config
from .models import TransformerModel
from .tokenizer import TokenizerManager
from .generation import TextGenerator
from .utils import setup_device, setup_logging, get_model_info


# Global model state
model_state = {
    'model': None,
    'tokenizer': None,
    'generator': None,
    'config': None,
    'loading': False
}


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str
    thinking: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[ChatMessage]
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    usage: Dict[str, int]
    model_info: Dict[str, str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    config_loaded: bool


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    parameters: str
    vocab_size: int
    max_sequence_length: int
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    print("Starting MinillM server...")
    yield
    # Shutdown
    print("Shutting down MinillM server...")


# Create FastAPI app
app = FastAPI(
    title="MinillM API",
    description="REST API for MinillM language model",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static file serving
web_static_path = Path(__file__).parent.parent / "web" / "static"
if web_static_path.exists():
    app.mount("/static", StaticFiles(directory=str(web_static_path)), name="static")


def load_model(config: Config) -> None:
    """Load model and tokenizer."""
    if model_state['loading']:
        return
    
    model_state['loading'] = True
    
    try:
        print("Loading tokenizer...")
        tokenizer = TokenizerManager(config.paths, config.tokens)
        
        print("Loading model...")
        device = setup_device(config.compute.device)
        model = TransformerModel.from_pretrained(config.paths.model_file, config.model).to(device)
        
        # Apply optimizations
        if config.performance.compile_model:
            print("Compiling model...")
            model.compile_model()
        
        print("Initializing generator...")
        generator = TextGenerator(model, tokenizer, config.generation)
        
        # Update global state
        model_state.update({
            'model': model,
            'tokenizer': tokenizer,
            'generator': generator,
            'config': config,
            'loading': False
        })
        
        print("Model loaded successfully!")
        
    except Exception as e:
        model_state['loading'] = False
        raise RuntimeError(f"Failed to load model: {e}")


@app.get("/")
async def read_root():
    """Serve the main web interface."""
    web_static_path = Path(__file__).parent.parent / "web" / "static"
    index_path = web_static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {"message": "MinillM API is running. Web interface not found."}


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    web_static_path = Path(__file__).parent.parent / "web" / "static"
    favicon_path = web_static_path / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    else:
        raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_state['model'] is not None else "loading",
        model_loaded=model_state['model'] is not None,
        config_loaded=model_state['config'] is not None
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    if model_state['model'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = model_state['model']
    config = model_state['config']
    
    model_info = get_model_info(model)
    
    return ModelInfoResponse(
        model_name="MinillM",
        parameters=model_info['total_parameters_formatted'],
        vocab_size=config.model.vocab_size,
        max_sequence_length=config.model.max_seq_len,
        device=str(next(model.parameters()).device)
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for non-streaming responses."""
    if model_state['generator'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Use /chat/stream for streaming responses")
    
    try:
        # Convert request to conversation format
        conversation = [
            {
                'role': msg.role,
                'content': msg.content,
                **(({'thinking': msg.thinking} if msg.thinking else {}))
            }
            for msg in request.messages
        ]
        
        # Generate response
        start_time = time.time()
        response = model_state['generator'].generate_response(
            conversation,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        generation_time = time.time() - start_time
        
        # Calculate usage statistics
        input_tokens = len(model_state['tokenizer'].encode_conversation(conversation))
        output_tokens = len(model_state['tokenizer'].encode(response))
        
        return ChatResponse(
            response=response,
            usage={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'generation_time_ms': int(generation_time * 1000)
            },
            model_info={
                'model': 'MinillM',
                'version': '0.1.0'
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    if model_state['generator'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to conversation format
        conversation = [
            {
                'role': msg.role,
                'content': msg.content,
                **(({'thinking': msg.thinking} if msg.thinking else {}))
            }
            for msg in request.messages
        ]
        
        async def generate_stream():
            """Generate streaming response."""
            # For now, we'll simulate streaming by generating the full response
            # and yielding it in chunks. Real streaming would require async generation.
            response = model_state['generator'].generate_response(
                conversation,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
            
            # Stream response word by word
            words = response.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming generation failed: {e}")


@app.post("/generate")
async def generate_text(
    prompt: str,
    max_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None
):
    """Simple text generation endpoint."""
    if model_state['generator'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create single-turn conversation
        conversation = [{'role': 'user', 'content': prompt}]
        
        response = model_state['generator'].generate_response(
            conversation,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        return {'generated_text': response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (useful for development)."""
    if model_state['config'] is None:
        raise HTTPException(status_code=400, detail="No config available for reload")
    
    # Clear current model
    model_state.update({
        'model': None,
        'tokenizer': None,
        'generator': None
    })
    
    # Reload model in background
    background_tasks.add_task(load_model, model_state['config'])
    
    return {'message': 'Model reload initiated'}


def start_server(config: Config):
    """Start the FastAPI server."""
    # Setup logging
    setup_logging(config.logging)
    
    # Load model
    load_model(config)
    
    # Store config in global state
    model_state['config'] = config
    
    # Start server
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        access_log=True
    )


if __name__ == "__main__":
    # For development - load default config
    from .config import load_config
    config = load_config()
    start_server(config)