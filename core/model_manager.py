"""
Smart Hybrid Model Manager with Dynamic VRAM-based Loading Strategy

Dynamically switches between:
- Parallel Mode: Multiple models loaded simultaneously when VRAM permits
- Serial Mode: Emergency unloading when VRAM is insufficient

Key Features:
- Real-time VRAM monitoring across all GPUs
- Capacity-aware loading with safety thresholds
- Automatic fallback to serial loading under memory pressure
- Robust error recovery with retry logic
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
from typing import Optional, Tuple, Dict


# Model VRAM estimates (in GB) with safety margins
# These are conservative estimates including overhead
MODEL_ESTIMATES = {
    'llava': 16.0,   # LLaVA-1.5-7B in float16
    'urm': 18.0,     # URM-LLaMa-3.1-8B in float16
    'clip': 3.0      # CLIP-ViT-Large in float16
}


class ModelManager:
    """
    Smart Hybrid Model Manager with VRAM-aware loading strategy.
    
    Architecture:
    - Cache: Stores currently loaded models
    - VRAM Monitor: Real-time memory tracking
    - Smart Loader: Decides parallel vs serial mode dynamically
    - Error Recovery: Automatic retry with full cleanup
    
    Loading Decision Tree:
    1. Check cache → return if already loaded
    2. Check VRAM → ensure sufficient space
    3. Load model with device_map="auto" for multi-GPU
    4. On failure → emergency cleanup and retry
    """
    
    def __init__(self, llava_path, urm_path, clip_path):
        """
        Initialize Smart Hybrid Model Manager.
        
        Args:
            llava_path: Path to LLaVA model
            urm_path: Path to URM model
            clip_path: Path to CLIP model
        """
        self.llava_path = llava_path
        self.urm_path = urm_path
        self.clip_path = clip_path
        
        # Model cache: stores loaded models
        self.model_cache = {}
        self.processor_cache = {}
        
        # Track loading mode for statistics
        self.parallel_loads = 0
        self.serial_loads = 0
        
        # Initialize VRAM monitoring
        self._init_vram_monitoring()
        
        print("[ModelManager] Smart Hybrid Loading initialized")
        self._print_vram_status()
    
    def _init_vram_monitoring(self):
        """Initialize VRAM monitoring and detect GPU configuration."""
        self.n_gpus = torch.cuda.device_count()
        
        if self.n_gpus == 0:
            raise RuntimeError("No CUDA GPUs detected! This system requires GPU.")
        
        print(f"[ModelManager] Detected {self.n_gpus} GPU(s)")
        
        # Print individual GPU info
        for i in range(self.n_gpus):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({total_gb:.1f}GB total)")
    
    def _get_total_free_vram(self) -> float:
        """
        Calculate total free VRAM across all visible GPUs.
        
        Returns:
            Total free VRAM in GB
        """
        total_free_bytes = 0
        
        for i in range(self.n_gpus):
            try:
                # Get free and total memory for this GPU
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                total_free_bytes += free_bytes
            except Exception as e:
                print(f"[ModelManager] Warning: Could not query GPU {i}: {e}")
        
        total_free_gb = total_free_bytes / (1024**3)
        return total_free_gb
    
    def _get_per_gpu_vram(self) -> Dict[int, Tuple[float, float]]:
        """
        Get detailed VRAM info for each GPU.
        
        Returns:
            Dictionary mapping GPU index to (free_gb, total_gb)
        """
        gpu_info = {}
        
        for i in range(self.n_gpus):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                gpu_info[i] = (free_gb, total_gb)
            except Exception as e:
                print(f"[ModelManager] Warning: Could not query GPU {i}: {e}")
                gpu_info[i] = (0.0, 0.0)
        
        return gpu_info
    
    def _print_vram_status(self):
        """Print current VRAM status across all GPUs."""
        gpu_info = self._get_per_gpu_vram()
        total_free = sum(free for free, _ in gpu_info.values())
        total_capacity = sum(total for _, total in gpu_info.values())
        
        print(f"[ModelManager] VRAM Status:")
        for gpu_id, (free_gb, total_gb) in gpu_info.items():
            used_gb = total_gb - free_gb
            usage_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
            print(f"  GPU {gpu_id}: {used_gb:.1f}/{total_gb:.1f} GB used ({usage_pct:.1f}%)")
        
        print(f"  Total Free: {total_free:.1f} GB / {total_capacity:.1f} GB")
        print(f"  Currently Loaded: {list(self.model_cache.keys())}")
    
    def _ensure_vram_space(self, model_key: str):
        """
        Ensure sufficient VRAM is available before loading a model.
        
        Smart Decision Logic:
        - If enough VRAM available → Parallel mode (do nothing)
        - If insufficient VRAM → Serial mode (emergency unload)
        
        Args:
            model_key: Key identifying the model ('llava', 'urm', 'clip')
        """
        required_gb = MODEL_ESTIMATES[model_key]
        available_gb = self._get_total_free_vram()
        
        print(f"[ModelManager] Checking VRAM for {model_key}:")
        print(f"  Required: ~{required_gb:.1f} GB")
        print(f"  Available: {available_gb:.1f} GB")
        
        # Condition A: Parallel Mode - Enough VRAM
        if available_gb > required_gb:
            print(f"  ✓ Parallel Mode: Sufficient VRAM ({available_gb:.1f} > {required_gb:.1f} GB)")
            self.parallel_loads += 1
            return
        
        # Condition B: Serial Fallback - Insufficient VRAM
        print(f"  ⚠ Serial Mode: Insufficient VRAM ({available_gb:.1f} < {required_gb:.1f} GB)")
        print(f"  → Triggering Emergency Unload...")
        
        # Emergency unload all other models
        models_to_unload = [key for key in self.model_cache.keys() if key != model_key]
        
        if models_to_unload:
            print(f"  → Unloading: {models_to_unload}")
            for key in models_to_unload:
                self._unload_model(key)
            
            # Verify space is now available
            available_after = self._get_total_free_vram()
            print(f"  → VRAM after cleanup: {available_after:.1f} GB")
            
            if available_after < required_gb:
                print(f"  ⚠ Warning: Still insufficient VRAM after cleanup!")
                print(f"  → Will attempt load anyway (device_map='auto' may help)")
        else:
            print(f"  → No other models to unload")
        
        self.serial_loads += 1
    
    def _unload_model(self, model_key: str):
        """
        Unload a specific model from cache and free VRAM.
        
        Args:
            model_key: Key identifying the model to unload
        """
        if model_key not in self.model_cache:
            return
        
        print(f"[ModelManager] Unloading {model_key}...")
        
        # Delete model and processor
        if model_key in self.model_cache:
            del self.model_cache[model_key]
        if model_key in self.processor_cache:
            del self.processor_cache[model_key]
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        print(f"[ModelManager] {model_key} unloaded successfully")
    
    def unload_all(self):
        """
        Emergency unload all models and free all VRAM.
        Used for error recovery.
        """
        print("[ModelManager] Emergency: Unloading ALL models...")
        
        model_keys = list(self.model_cache.keys())
        
        for key in model_keys:
            self._unload_model(key)
        
        # Extra aggressive cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            for i in range(self.n_gpus):
                torch.cuda.synchronize(i)
        
        print("[ModelManager] All models unloaded")
        self._print_vram_status()
    
    def load_llava(self) -> Tuple:
        """
        Load LLaVA model with smart VRAM management.
        
        Returns:
            Tuple of (model, processor)
        """
        model_key = 'llava'
        
        # Check cache first
        if model_key in self.model_cache:
            print(f"[ModelManager] LLaVA already loaded (cache hit)")
            return self.model_cache[model_key], self.processor_cache[model_key]
        
        # Ensure sufficient VRAM
        self._ensure_vram_space(model_key)
        
        print(f"[ModelManager] Loading LLaVA from {self.llava_path}...")
        
        try:
            # Load model with device_map="auto" for multi-GPU distribution
            model = LlavaForConditionalGeneration.from_pretrained(
                self.llava_path,
                torch_dtype=torch.float16,
                device_map="auto",  # Automatic multi-GPU distribution
                trust_remote_code=True
            )
            
            processor = AutoProcessor.from_pretrained(
                self.llava_path,
                trust_remote_code=True
            )
            
            # Cache the loaded model
            self.model_cache[model_key] = model
            self.processor_cache[model_key] = processor
            
            print("[ModelManager] LLaVA loaded successfully")
            self._print_vram_status()
            
            return model, processor
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[ModelManager] OOM Error during LLaVA load: {e}")
                print("[ModelManager] Attempting recovery: Full cleanup + retry...")
                
                # Emergency cleanup
                self.unload_all()
                
                # Retry with fresh VRAM
                print("[ModelManager] Retrying LLaVA load after cleanup...")
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
                
                self.model_cache[model_key] = model
                self.processor_cache[model_key] = processor
                
                print("[ModelManager] LLaVA loaded successfully after retry")
                return model, processor
            else:
                raise
    
    def load_urm(self) -> Tuple:
        """
        Load URM model with smart VRAM management.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_key = 'urm'
        
        # Check cache first
        if model_key in self.model_cache:
            print(f"[ModelManager] URM already loaded (cache hit)")
            return self.model_cache[model_key], self.processor_cache[model_key]
        
        # Ensure sufficient VRAM
        self._ensure_vram_space(model_key)
        
        print(f"[ModelManager] Loading URM from {self.urm_path}...")
        
        try:
            # Load model with device_map="auto"
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
            
            # Cache the loaded model
            self.model_cache[model_key] = model
            self.processor_cache[model_key] = tokenizer
            
            print("[ModelManager] URM loaded successfully")
            self._print_vram_status()
            
            return model, tokenizer
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[ModelManager] OOM Error during URM load: {e}")
                print("[ModelManager] Attempting recovery: Full cleanup + retry...")
                
                # Emergency cleanup
                self.unload_all()
                
                # Retry
                print("[ModelManager] Retrying URM load after cleanup...")
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
                
                self.model_cache[model_key] = model
                self.processor_cache[model_key] = tokenizer
                
                print("[ModelManager] URM loaded successfully after retry")
                return model, tokenizer
            else:
                raise
    
    def load_clip(self) -> Tuple:
        """
        Load CLIP model with smart VRAM management.
        
        Returns:
            Tuple of (model, processor)
        """
        model_key = 'clip'
        
        # Check cache first
        if model_key in self.model_cache:
            print(f"[ModelManager] CLIP already loaded (cache hit)")
            return self.model_cache[model_key], self.processor_cache[model_key]
        
        # Ensure sufficient VRAM
        self._ensure_vram_space(model_key)
        
        print(f"[ModelManager] Loading CLIP from {self.clip_path}...")
        
        try:
            # CLIP is small, but still use device_map for consistency
            model = CLIPModel.from_pretrained(
                self.clip_path,
                torch_dtype=torch.float16
            )
            
            # For CLIP, we manually move to first GPU if not using device_map
            if self.n_gpus > 0:
                model = model.to("cuda:0")
            
            processor = CLIPProcessor.from_pretrained(self.clip_path)
            
            # Cache the loaded model
            self.model_cache[model_key] = model
            self.processor_cache[model_key] = processor
            
            print("[ModelManager] CLIP loaded successfully")
            self._print_vram_status()
            
            return model, processor
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[ModelManager] OOM Error during CLIP load: {e}")
                print("[ModelManager] Attempting recovery: Full cleanup + retry...")
                
                # Emergency cleanup
                self.unload_all()
                
                # Retry
                print("[ModelManager] Retrying CLIP load after cleanup...")
                model = CLIPModel.from_pretrained(
                    self.clip_path,
                    torch_dtype=torch.float16
                ).to("cuda:0")
                
                processor = CLIPProcessor.from_pretrained(self.clip_path)
                
                self.model_cache[model_key] = model
                self.processor_cache[model_key] = processor
                
                print("[ModelManager] CLIP loaded successfully after retry")
                return model, processor
            else:
                raise
    
    def unload_model(self, model_name: Optional[str] = None):
        """
        Unload a specific model or all models.
        
        Args:
            model_name: Name of model to unload ('llava', 'urm', 'clip'),
                       or None to unload all
        """
        if model_name is None:
            self.unload_all()
        else:
            self._unload_model(model_name)
    
    def get_statistics(self) -> Dict:
        """
        Get model manager statistics.
        
        Returns:
            Dictionary with loading statistics and VRAM info
        """
        gpu_info = self._get_per_gpu_vram()
        total_free = sum(free for free, _ in gpu_info.values())
        
        return {
            'parallel_loads': self.parallel_loads,
            'serial_loads': self.serial_loads,
            'parallel_ratio': self.parallel_loads / max(self.parallel_loads + self.serial_loads, 1),
            'currently_loaded': list(self.model_cache.keys()),
            'n_models_cached': len(self.model_cache),
            'total_free_vram_gb': total_free,
            'gpu_info': {
                f'gpu_{i}': {'free_gb': free, 'total_gb': total}
                for i, (free, total) in gpu_info.items()
            }
        }
    
    def print_statistics(self):
        """Print comprehensive model manager statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("MODEL MANAGER STATISTICS")
        print("="*80)
        print(f"Loading Strategy:")
        print(f"  Parallel loads: {stats['parallel_loads']}")
        print(f"  Serial loads:   {stats['serial_loads']}")
        print(f"  Parallel ratio: {stats['parallel_ratio']:.1%}")
        print(f"\nCache Status:")
        print(f"  Models loaded: {stats['currently_loaded']}")
        print(f"  Cache size:    {stats['n_models_cached']}")
        print(f"\nVRAM Status:")
        print(f"  Total free: {stats['total_free_vram_gb']:.1f} GB")
        for gpu_name, gpu_data in stats['gpu_info'].items():
            print(f"  {gpu_name}: {gpu_data['free_gb']:.1f}/{gpu_data['total_gb']:.1f} GB free")
        print("="*80 + "\n")
