"""
Web server for MinillM using FastAPI.
Provides REST API endpoints for chat and generation.
"""

import asyncio
import time
from typing import List, Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import Config
if HAS_TORCH:
    try:
        from .models import TransformerModel
        from .tokenizer import TokenizerManager
        from .generation import TextGenerator
        from .model_loader import ModelManager
        from .utils import setup_device, setup_logging, get_model_info
        HAS_MODELS = True
    except ImportError as e:
        print(f"Warning: Some model modules not available: {e}")
        HAS_MODELS = False
else:
    HAS_MODELS = False


# Global model state
model_state = {
    'model': None,
    'tokenizer': None,
    'generator': None,
    'config': None,
    'model_manager': None,
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


class ModelSwitchRequest(BaseModel):
    """Model switch request."""
    model_name: str


class ModelSwitchResponse(BaseModel):
    """Model switch response."""
    success: bool
    message: str
    model_name: Optional[str] = None


class AvailableModelsResponse(BaseModel):
    """Available models response."""
    models: List[Dict[str, str]]


class CurrentModelResponse(BaseModel):
    """Current model response."""
    name: str
    type: str
    loaded: bool
    description: str


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
    """Load model and tokenizer using the new model manager."""
    if model_state['loading']:
        return
    
    model_state['loading'] = True
    
    try:
        if not HAS_MODELS:
            print("Warning: Model loading dependencies not available")
            model_state.update({
                'model': None,
                'tokenizer': None,
                'generator': None,
                'model_manager': None,
                'config': config,
                'loading': False
            })
            return
        
        print("Initializing model manager...")
        model_manager = ModelManager(config)
        
        print(f"Loading model: {config.model_source.name}")
        model, tokenizer = model_manager.load_model(config.model_source)
        
        # Create generator for local models
        generator = None
        if config.model_source.type == "local":
            generator = TextGenerator(model, tokenizer, config.generation)
        
        # Update global state
        model_state.update({
            'model': model,
            'tokenizer': tokenizer,
            'generator': generator,
            'model_manager': model_manager,
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
    if model_state['model_manager'] is None:
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
        
        # Generate response using model manager
        start_time = time.time()
        
        # For local models, use the existing generator
        if model_state['config'].model_source.type == "local" and model_state['generator']:
            response = model_state['generator'].generate_response(
                conversation,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        else:
            # For other models, use the model manager
            # Convert conversation to simple prompt
            prompt = conversation[-1]['content'] if conversation else ""
            response = model_state['model_manager'].generate(
                prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        
        generation_time = time.time() - start_time
        
        # Calculate usage statistics (simplified for non-local models)
        try:
            if model_state['tokenizer'] and hasattr(model_state['tokenizer'], 'encode_conversation'):
                input_tokens = len(model_state['tokenizer'].encode_conversation(conversation))
                output_tokens = len(model_state['tokenizer'].encode(response))
            else:
                # Rough estimation for other models
                input_tokens = len(conversation[-1]['content'].split()) if conversation else 0
                output_tokens = len(response.split())
        except:
            input_tokens = len(conversation[-1]['content'].split()) if conversation else 0
            output_tokens = len(response.split())
        
        return ChatResponse(
            response=response,
            usage={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'generation_time_ms': int(generation_time * 1000)
            },
            model_info={
                'model': model_state['config'].model_source.name,
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


@app.get("/models/available")
async def get_available_models():
    """Get list of available models."""
    if not HAS_MODELS:
        # Return a basic model list when models aren't available
        return [
            {"name": "Demo Mode", "type": "demo", "description": "Demo mode - model dependencies not available"}
        ]
    
    if model_state['model_manager'] is None:
        # Return available models from config if manager isn't initialized yet
        if model_state['config'] and hasattr(model_state['config'], 'available_models'):
            return [
                {
                    "name": model.name,
                    "type": model.type,
                    "description": f"{model.type.title()} model: {model.name}"
                }
                for model in model_state['config'].available_models
            ]
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        models = model_state['model_manager'].get_available_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {e}")


@app.get("/models/current", response_model=CurrentModelResponse)
async def get_current_model():
    """Get information about the current model."""
    if not HAS_MODELS:
        return CurrentModelResponse(
            name="Demo Mode",
            type="demo",
            loaded=False,
            description="Demo mode - model dependencies not available"
        )
    
    if model_state['model_manager'] is None:
        # Return current model from config if manager isn't initialized yet
        if model_state['config'] and hasattr(model_state['config'], 'model_source'):
            return CurrentModelResponse(
                name=model_state['config'].model_source.name,
                type=model_state['config'].model_source.type,
                loaded=False,
                description=f"{model_state['config'].model_source.type.title()} model: {model_state['config'].model_source.name} (not loaded)"
            )
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        model_info = model_state['model_manager'].get_current_model_info()
        return CurrentModelResponse(
            name=model_info['name'],
            type=model_info['type'],
            loaded=model_info['loaded'],
            description=f"{model_info['type'].title()} model: {model_info['name']}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current model info: {e}")


@app.post("/models/switch", response_model=ModelSwitchResponse)
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model."""
    if model_state['model_manager'] is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        success = model_state['model_manager'].switch_model(request.model_name)
        
        if success:
            # Update global state
            model_state.update({
                'model': model_state['model_manager'].current_model,
                'tokenizer': model_state['model_manager'].current_tokenizer,
                'generator': None  # Will be recreated if needed
            })
            
            # Create generator for local models
            if model_state['config'].model_source.type == "local":
                from .generation import TextGenerator
                model_state['generator'] = TextGenerator(
                    model_state['model'], 
                    model_state['tokenizer'], 
                    model_state['config'].generation
                )
            
            return ModelSwitchResponse(
                success=True,
                message=f"Successfully switched to {request.model_name}",
                model_name=request.model_name
            )
        else:
            return ModelSwitchResponse(
                success=False,
                message=f"Failed to switch to {request.model_name}"
            )
    except Exception as e:
        return ModelSwitchResponse(
            success=False,
            message=f"Error switching model: {e}"
        )


@app.post("/models/upload")
async def upload_model(model_file: UploadFile = File(...)):
    """Upload and load a local model file."""
    if not HAS_MODELS:
        return {"success": False, "message": "Model loading dependencies not available"}
    
    try:
        # Validate file type
        allowed_extensions = ['.pt', '.pth', '.bin', '.safetensors']
        if not any(model_file.filename.endswith(ext) for ext in allowed_extensions):
            return {"success": False, "message": "Invalid file type. Supported: .pt, .pth, .bin, .safetensors"}
        
        # Create uploads directory
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = uploads_dir / model_file.filename
        with open(file_path, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Try to load the model
        if model_state['model_manager'] is None:
            return {"success": False, "message": "Model manager not initialized"}
        
        # Create a new model source config for the uploaded file
        from .config import ModelSourceConfig
        new_model_source = ModelSourceConfig(
            type="local",
            name=model_file.filename,
            path=str(file_path)
        )
        
        # Try to load the model
        success = model_state['model_manager'].switch_model_source(new_model_source)
        
        if success:
            # Update global state
            model_state.update({
                'model': model_state['model_manager'].current_model,
                'tokenizer': model_state['model_manager'].current_tokenizer,
                'generator': None
            })
            
            # Create generator for local models
            if model_state['config'] and HAS_MODELS:
                try:
                    from .generation import TextGenerator
                    model_state['generator'] = TextGenerator(
                        model_state['model'], 
                        model_state['tokenizer'], 
                        model_state['config'].generation
                    )
                except Exception as e:
                    print(f"Warning: Could not create generator: {e}")
            
            return {"success": True, "message": f"Successfully loaded {model_file.filename}"}
        else:
            # Clean up file if loading failed
            file_path.unlink(missing_ok=True)
            return {"success": False, "message": "Failed to load the uploaded model"}
            
    except Exception as e:
        return {"success": False, "message": f"Error uploading model: {str(e)}"}


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (useful for development)."""
    if model_state['config'] is None:
        raise HTTPException(status_code=400, detail="No config available for reload")
    
    # Clear current model
    model_state.update({
        'model': None,
        'tokenizer': None,
        'generator': None,
        'model_manager': None
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