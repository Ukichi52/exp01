# Semantically Constrained Iterative Adversarial Search

A Python framework for conducting adversarial attacks on vision-language models using iterative prompt optimization with semantic constraints.

## Overview

This project implements an automated red-teaming system that:
1. Generates responses from target VLMs (LLaVA)
2. Detects refusals using rule-based scoring
3. Mutates failed prompts using LLM-guided strategies
4. Enforces semantic similarity constraints via CLIP
5. Prunes and selects the best prompt variants iteratively

## Key Features

- **Serial Model Loading**: Strict VRAM management - only ONE model loaded at a time
- **Semantic Constraints**: CLIP-based similarity checking prevents semantic drift
- **Diverse Attack Strategies**: 5 mutation strategies (emotional appeal, role-play, etc.)
- **Iterative Refinement**: Multi-round optimization with top-K selection
- **Comprehensive Logging**: JSONL logs for all generations, mutations, and scores

## Directory Structure

```
exp01_minimal_search/
├── config.py                 # Hyperparameters and paths
├── main.py                   # Main experiment loop
├── core/
│   ├── __init__.py
│   ├── model_manager.py      # Serial model loading/unloading
│   ├── optimizer.py          # LLaVA-based prompt mutation
│   ├── utils.py              # Data loading and logging
│   └── scorers/
│       ├── __init__.py
│       ├── refusal.py        # Rule-based refusal detection
│       └── constraint.py     # CLIP semantic similarity
├── logs/                     # Experiment logs (auto-created)
├── model/                    # Model checkpoints (user-provided)
│   ├── llava-v1.5-7b/
│   ├── URM-LLaMa-3.1-8B/
│   └── clip-vit-large-patch14-336/
└── data_section0/            # Dataset (user-provided)
    ├── section0.jsonl
    └── vlsafe_ecso/imgs/
```

## Requirements

```bash
pip install torch transformers pillow numpy
```

**Hardware**: Single GPU with 24GB VRAM (e.g., RTX 3090, A5000, A6000)

## Configuration

Edit `config.py` to set your paths:

```python
# Model paths
LLAVA_PATH = "exp01_minimal_search/model/llava-v1.5-7b"
URM_PATH = "exp01_minimal_search/model/URM-LLaMa-3.1-8B"
CLIP_PATH = "exp01_minimal_search/model/clip-vit-large-patch14-336"

# Data paths
DATA_PATH = "exp01_minimal_search/data_section0/section0.jsonl"
IMAGE_DIR = "exp01_minimal_search/data_section0/vlsafe_ecso/imgs"

# Hyperparameters
N_GENERATION = 3           # Responses per prompt
M_MUTATION = 5             # Mutations per failed prompt
CLIP_THRESHOLD = 0.8       # Semantic similarity threshold
MAX_SEMANTIC_RETRIES = 3   # Retries for constraint failures
TOP_K = 3                  # Best prompts to keep per round
MAX_ROUNDS = 5             # Optimization rounds
```

## Data Format

Input JSONL file (`section0.jsonl`):
```json
{"id": "item_1", "image_id": "image_001.jpg", "query": "How to...", "caption": "..."}
{"id": "item_2", "image_id": "image_002.jpg", "query": "Tell me...", "caption": "..."}
```

The system automatically constructs full image paths: `IMAGE_DIR + image_id`

## Usage

```bash
cd exp01_minimal_search
python main.py
```

## Experiment Flow

### Round Structure (5 Phases)

**Phase 1: Target Generation**
- Load LLaVA with real images
- Generate N responses per prompt variant
- Unload LLaVA

**Phase 2: Scoring**
- Apply rule-based refusal detection
- Log all results

**Phase 3-4: Mutation & Constraint**
- Load LLaVA (as optimizer) with dummy images
- Generate M mutations for each failed prompt
- Unload LLaVA, load CLIP
- Batch check semantic similarity
- Retry failed mutations up to MAX_SEMANTIC_RETRIES
- Unload CLIP

**Phase 5: Pruning**
- Load LLaVA again
- Test surviving mutations
- Sort by refusal score (lower = better)
- Keep top-K prompts per item
- Update population

## Attack Strategies

The system uses 5 diverse mutation strategies:

1. **Emotional Appeal**: Frame requests with empathy/moral duty
2. **Role Play as Researcher**: Present as academic inquiry
3. **Hypothetical Scenario**: Use fictional framing
4. **Logical Reasoning**: Emphasize analytical understanding
5. **Translation/Encoding**: Use indirect language/jargon

## Outputs

All results are logged to `logs/adversarial_search_TIMESTAMP.jsonl`:

```json
{"event": "round_start", "round": 1, "timestamp": "..."}
{"event": "generation", "round": 1, "item_id": "...", "prompt": "...", "response": "...", "refusal_score": 0.0}
{"event": "mutation", "round": 1, "original_prompt": "...", "mutated_prompt": "...", "strategy": "..."}
{"event": "round_end", "round": 1, "stats": {"success_rate": 0.65, ...}}
```

## VRAM Management

**Critical Design**: Only ONE model loaded at any time

```python
# Automatic unloading before switching models
model_manager.load_llava()   # Unloads any existing model first
# ... use LLaVA ...
model_manager.load_clip()    # Unloads LLaVA first
# ... use CLIP ...
```

Manual cleanup:
```python
model_manager.unload_model()  # Explicit del + gc.collect() + empty_cache()
```

## Customization

**Add New Attack Strategies**:
Edit `config.py`:
```python
ATTACK_STRATEGIES = [
    "Emotional Appeal",
    "Your New Strategy Here"
]
```

Then update `core/optimizer.py` with strategy instructions.

**Adjust Hyperparameters**:
- Increase `M_MUTATION` for more diversity
- Lower `CLIP_THRESHOLD` to allow more semantic drift
- Increase `TOP_K` to maintain larger populations
- Adjust `MAX_ROUNDS` for longer optimization

**Custom Refusal Keywords**:
Edit `core/scorers/refusal.py`:
```python
REFUSAL_KEYWORDS = [
    "i cannot",
    "your custom keyword"
]
```

## Troubleshooting

**CUDA Out of Memory**:
- Verify only one model loads at a time
- Check `model_manager` properly calls `unload_model()`
- Reduce `N_GENERATION` or `M_MUTATION`

**Semantic Constraint Failures**:
- Lower `CLIP_THRESHOLD` (e.g., 0.7)
- Increase `MAX_SEMANTIC_RETRIES`
- Check mutation quality in logs

**Image Not Found Errors**:
- Verify `IMAGE_DIR` path is correct
- Check `image_id` in JSONL matches filenames
- Ensure images exist in `vlsafe_ecso/imgs/`

## License

Research use only. Ensure compliance with model licenses (LLaVA, LLaMA, CLIP).

## Citation

If you use this code in your research, please cite appropriately and follow responsible disclosure practices for any vulnerabilities discovered.