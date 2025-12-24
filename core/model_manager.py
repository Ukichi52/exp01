"""
Model Manager for Serial Model Loading
Ensures only ONE model is loaded at a time to respect 24GB VRAM constraint.
"""

import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
    AutoImageProcessor
)
from PIL import Image


class ModelManager:
    """
    Manages loading and unloading of LLaVA, URM, and CLIP models.
    Enforces strict serial loading to prevent VRAM overflow.
    """
    
    def __init__(self, llava_path, urm_path, clip_path):
        self.llava_path = llava_path
        self.urm_path = urm_path
        self.clip_path = clip_path
        
        self.current_model = None
        self.current_processor = None
        self.current_model_name = None
        
        print("[ModelManager] Initialized with serial loading policy")
    
    def unload_model(self):
        """
        Explicitly unload current model and free VRAM.
        CRITICAL: Must be called before loading a different model.
        """
        if self.current_model is not None:
            print(f"[ModelManager] Unloading {self.current_model_name}...")
            
            # Delete model and processor
            del self.current_model
            del self.current_processor
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            self.current_model = None
            self.current_processor = None
            self.current_model_name = None
            
            print("[ModelManager] VRAM cleared successfully")
    
    def load_llava(self):
        """
        Load LLaVA model for vision-language tasks.
        Returns: (model, processor)
        """
        if self.current_model_name == "llava":
            print("[ModelManager] LLaVA already loaded, reusing...")
            return self.current_model, self.current_processor
        
        # Unload any existing model first
        if self.current_model_name is not None:
            self.unload_model()
        
        print(f"[ModelManager] Loading LLaVA from {self.llava_path}...")
        
        # Use LlavaForConditionalGeneration instead of AutoModelForCausalLM
        model = LlavaForConditionalGeneration.from_pretrained(
            self.llava_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            self.llava_path,
            trust_remote_code=True
        )
        
        self.current_model = model
        self.current_processor = processor
        self.current_model_name = "llava"
        
        print("[ModelManager] LLaVA loaded successfully")
        return model, processor
    
    def load_urm(self):
        """
        Load URM model for refusal detection.
        Returns: (model, tokenizer)
        """
        if self.current_model_name == "urm":
            print("[ModelManager] URM already loaded, reusing...")
            return self.current_model, self.current_processor
        
        # Unload any existing model first
        if self.current_model_name is not None:
            self.unload_model()
        
        print(f"[ModelManager] Loading URM from {self.urm_path}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.urm_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.urm_path,
            trust_remote_code=True
        )
        
        self.current_model = model
        self.current_processor = tokenizer
        self.current_model_name = "urm"
        
        print("[ModelManager] URM loaded successfully")
        return model, tokenizer
    
    def load_clip(self):
        """
        Load CLIP model for semantic similarity checks.
        Returns: (model, processor)
        """
        if self.current_model_name == "clip":
            print("[ModelManager] CLIP already loaded, reusing...")
            return self.current_model, self.current_processor
        
        # Unload any existing model first
        if self.current_model_name is not None:
            self.unload_model()
        
        print(f"[ModelManager] Loading CLIP from {self.clip_path}...")
        
        model = CLIPModel.from_pretrained(
            self.clip_path,
            torch_dtype=torch.float16
        ).to("cuda")
        
        processor = CLIPProcessor.from_pretrained(self.clip_path)
        
        self.current_model = model
        self.current_processor = processor
        self.current_model_name = "clip"
        
        print("[ModelManager] CLIP loaded successfully")
        return model, processor
