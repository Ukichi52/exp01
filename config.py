"""
Configuration file for Semantically Constrained Iterative Adversarial Search
Defines all hyperparameters, paths, and attack strategies.
"""

import os
GPU_ID = "7"
# ============================================================================
# MODEL PATHS
# ============================================================================
LLAVA_PATH = "/data/heyuji/exp01_minimal_search/model/llava-v1.5-7b"
URM_PATH = "/data/heyuji/exp01_minimal_search/model/URM-LLaMa-3.1-8B"
CLIP_PATH = "/data/heyuji/exp01_minimal_search/model/clip-vit-large-patch14-336"

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_PATH = "/data/heyuji/exp01_minimal_search/data_section0/vlsafe_ecso/query20.jsonl"
IMAGE_DIR = "/data/heyuji/exp01_minimal_search/data_section0/vlsafe_ecso/imgs"

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
N_GENERATION = 3          # Number of responses to generate per query
M_MUTATION = 5            # Number of mutations to generate per failed query
CLIP_THRESHOLD = 0.8      # Minimum cosine similarity for semantic constraint
MAX_SEMANTIC_RETRIES = 3  # Maximum retries for failed semantic checks
TOP_K = 3                 # Number of best prompts to keep per round
MAX_ROUNDS = 5            # Maximum number of optimization rounds

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
MAX_NEW_TOKENS = 2048     # Maximum number of optimization rounds
TEMPERATURE = 0.7
TOP_P = 0.9

# ============================================================================
# ATTACK STRATEGIES
# ============================================================================
ATTACK_STRATEGIES = [
    "Emotional Appeal",
    "Role Play as Researcher",
    "Hypothetical Scenario",
    "Logical Reasoning",
    "Translation/Encoding"
]

# ============================================================================
# LOGGING
# ============================================================================
LOG_DIR = "/data/heyuji/exp01_minimal_search/logs"
os.makedirs(LOG_DIR, exist_ok=True)