"""
Utility functions for data loading and split logging system.

Logging Architecture:
- _stats.jsonl: Round summaries, metrics, experiment-level events
- _mutations.jsonl: Optimizer decisions, prompt changes, bandit updates
- _generations.jsonl: Model inputs/outputs, refusal scores, jailbreak attempts
"""

import json
import os
from datetime import datetime
from pathlib import Path
import config


def load_dataset(jsonl_path):
    """
    Load dataset from JSONL file and construct full image paths.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        List of dictionaries with keys: id, prompt, caption, image_path
    """
    dataset = []
    
    print(f"[DataLoader] Loading dataset from {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                image_id = data.get('image_id', '')
                query = data.get('query', '')
                caption = data.get('caption', '')
                data_id = data.get('id', f'item_{line_num}')
                
                # Construct full image path
                image_path = os.path.join(config.IMAGE_DIR, image_id)
                
                # Verify image exists
                if not os.path.exists(image_path):
                    print(f"  [Warning] Image not found: {image_path}")
                    continue
                
                dataset.append({
                    'id': data_id,
                    'prompt': query,
                    'caption': caption,
                    'image_path': image_path,
                    'image_id': image_id
                })
                
            except json.JSONDecodeError as e:
                print(f"  [Error] Failed to parse line {line_num}: {e}")
                continue
    
    print(f"[DataLoader] Loaded {len(dataset)} items successfully")
    return dataset


class Logger:
    """
    Split Logging System for multi-file experiment tracking.
    
    Architecture:
    - _stats.jsonl: Experiment metadata, round summaries, aggregate metrics
    - _mutations.jsonl: Optimizer events, prompt mutations, bandit decisions
    - _generations.jsonl: Target model generations, responses, refusal scores
    
    Benefits:
    - Easier post-processing and analysis
    - Faster filtering by event type
    - Reduced file I/O contention
    - Better scalability for large experiments
    """
    
    def __init__(self, experiment_name="experiment"):
        """
        Initialize split logging system with three output files.
        
        Args:
            experiment_name: Base name for the experiment (timestamp added)
        """
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create base filename
        base_filename = f"{experiment_name}_{timestamp}"
        
        # Define three log files
        self.stats_file = os.path.join(config.LOG_DIR, f"{base_filename}_stats.jsonl")
        self.mutations_file = os.path.join(config.LOG_DIR, f"{base_filename}_mutations.jsonl")
        self.generations_file = os.path.join(config.LOG_DIR, f"{base_filename}_generations.jsonl")
        
        # Create log directory if it doesn't exist
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Log initialization
        print(f"[Logger] Split logging initialized:")
        print(f"  Stats:       {self.stats_file}")
        print(f"  Mutations:   {self.mutations_file}")
        print(f"  Generations: {self.generations_file}")
        
        # Write header information to stats file
        self._log_experiment_start()
    
    def _log_experiment_start(self):
        """Log experiment initialization metadata."""
        self._write_to_stats({
            'event': 'experiment_start',
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'N_GENERATION': config.N_GENERATION,
                'M_MUTATION': config.M_MUTATION,
                'CLIP_THRESHOLD': config.CLIP_THRESHOLD,
                'MAX_SEMANTIC_RETRIES': config.MAX_SEMANTIC_RETRIES,
                'TOP_K': config.TOP_K,
                'MAX_ROUNDS': config.MAX_ROUNDS
            }
        })
    
    def _write_to_stats(self, data):
        """Write to stats log file."""
        with open(self.stats_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def _write_to_mutations(self, data):
        """Write to mutations log file."""
        with open(self.mutations_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def _write_to_generations(self, data):
        """Write to generations log file."""
        with open(self.generations_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def log(self, data):
        """
        Generic log method (for backward compatibility).
        Routes to stats file by default.
        
        Args:
            data: Dictionary to log
        """
        self._write_to_stats(data)
    
    def log_round_start(self, round_num):
        """
        Log the start of an optimization round.
        Logged to: _stats.jsonl
        
        Args:
            round_num: The round number
        """
        self._write_to_stats({
            'event': 'round_start',
            'round': round_num,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_round_end(self, round_num, stats):
        """
        Log the end of a round with aggregate statistics.
        Logged to: _stats.jsonl
        
        Args:
            round_num: The round number
            stats: Dictionary containing round statistics
        """
        self._write_to_stats({
            'event': 'round_end',
            'round': round_num,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_generation(self, round_num, item_id, prompt, response, 
                       refusal_score, is_refused):
        """
        Log a target model generation attempt.
        Logged to: _generations.jsonl
        
        Args:
            round_num: Current round number
            item_id: Item identifier
            prompt: Input prompt to the model
            response: Model's response
            refusal_score: Computed refusal score (0.0 = success, 1.0 = refused)
            is_refused: Boolean indicating if response was a refusal
        """
        self._write_to_generations({
            'event': 'generation',
            'round': round_num,
            'item_id': item_id,
            'prompt': prompt,
            'response': response,
            'refusal_score': refusal_score,
            'is_refused': is_refused,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_mutation(self, round_num, item_id, original_prompt, 
                    mutated_prompt, strategy, passed_constraint):
        """
        Log a prompt mutation event from the optimizer.
        Logged to: _mutations.jsonl
        
        Args:
            round_num: Current round number
            item_id: Item identifier
            original_prompt: The original prompt before mutation
            mutated_prompt: The mutated prompt after optimization
            strategy: Strategy used for mutation (e.g., "OPTIMIZE_OBJECT")
            passed_constraint: Boolean indicating if semantic constraint was satisfied
        """
        self._write_to_mutations({
            'event': 'mutation',
            'round': round_num,
            'item_id': item_id,
            'original_prompt': original_prompt,
            'mutated_prompt': mutated_prompt,
            'strategy': strategy,
            'passed_constraint': passed_constraint,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_bandit_selection(self, round_num, strategy, theta_samples):
        """
        Log bandit strategy selection with Thompson Sampling details.
        Logged to: _mutations.jsonl
        
        Args:
            round_num: Current round number
            strategy: Selected strategy
            theta_samples: Dictionary of sampled theta values for each arm
        """
        self._write_to_mutations({
            'event': 'bandit_selection',
            'round': round_num,
            'selected_strategy': strategy,
            'theta_samples': theta_samples,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_bandit_update(self, round_num, strategy, is_success, alpha, beta):
        """
        Log bandit parameter update after receiving feedback.
        Logged to: _mutations.jsonl
        
        Args:
            round_num: Current round number
            strategy: Strategy that was used
            is_success: Whether the jailbreak succeeded
            alpha: Updated alpha parameter (success count)
            beta: Updated beta parameter (failure count)
        """
        self._write_to_mutations({
            'event': 'bandit_update',
            'round': round_num,
            'strategy': strategy,
            'is_success': is_success,
            'alpha': alpha,
            'beta': beta,
            'expected_reward': alpha / (alpha + beta),
            'timestamp': datetime.now().isoformat()
        })
    
    def log_bandit_statistics(self, round_num, statistics):
        """
        Log aggregate bandit statistics for the round.
        Logged to: _stats.jsonl
        
        Args:
            round_num: Current round number
            statistics: Dictionary of statistics for each strategy
        """
        self._write_to_stats({
            'event': 'bandit_statistics',
            'round': round_num,
            'statistics': statistics,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_semantic_constraint_check(self, round_num, item_id, 
                                      candidate, original, similarity, passed):
        """
        Log semantic constraint checking results.
        Logged to: _mutations.jsonl
        
        Args:
            round_num: Current round number
            item_id: Item identifier
            candidate: Candidate prompt being checked
            original: Original reference prompt
            similarity: CLIP similarity score
            passed: Whether constraint was satisfied
        """
        self._write_to_mutations({
            'event': 'semantic_constraint',
            'round': round_num,
            'item_id': item_id,
            'candidate': candidate,
            'original': original,
            'similarity': similarity,
            'passed': passed,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_phase_start(self, round_num, phase_name):
        """
        Log the start of a specific phase in the optimization loop.
        Logged to: _stats.jsonl
        
        Args:
            round_num: Current round number
            phase_name: Name of the phase (e.g., "target_generation", "mutation")
        """
        self._write_to_stats({
            'event': 'phase_start',
            'round': round_num,
            'phase': phase_name,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_phase_end(self, round_num, phase_name, duration_seconds, metrics=None):
        """
        Log the end of a phase with duration and optional metrics.
        Logged to: _stats.jsonl
        
        Args:
            round_num: Current round number
            phase_name: Name of the phase
            duration_seconds: Time taken for the phase
            metrics: Optional dictionary of phase-specific metrics
        """
        self._write_to_stats({
            'event': 'phase_end',
            'round': round_num,
            'phase': phase_name,
            'duration_seconds': duration_seconds,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def log_error(self, error_msg, context=None):
        """
        Log an error event.
        Logged to: _stats.jsonl
        
        Args:
            error_msg: Error message
            context: Optional context dictionary
        """
        self._write_to_stats({
            'event': 'error',
            'error': error_msg,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_aom_decomposition(self, round_num, item_id, original_query, 
                              action, obj, mechanism):
        """
        Log Action-Object-Mechanism decomposition.
        Logged to: _mutations.jsonl
        
        Args:
            round_num: Current round number
            item_id: Item identifier
            original_query: Original query string
            action: Extracted action component
            obj: Extracted object component
            mechanism: Extracted mechanism component
        """
        self._write_to_mutations({
            'event': 'aom_decomposition',
            'round': round_num,
            'item_id': item_id,
            'original_query': original_query,
            'components': {
                'action': action,
                'object': obj,
                'mechanism': mechanism
            },
            'timestamp': datetime.now().isoformat()
        })
    
    def close(self):
        """
        Finalize logging and write experiment summary.
        """
        self._write_to_stats({
            'event': 'experiment_end',
            'timestamp': datetime.now().isoformat()
        })
        print(f"[Logger] Experiment logs finalized")


def format_duration(seconds):
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_progress_bar(iteration, total, prefix='', suffix='', 
                       length=50, fill='█'):
    """
    Print a progress bar to console.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
        fill: Bar fill character
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    
    if iteration == total:
        print()  # New line on completion


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def load_split_logs(base_path):
    """
    Load all three log files from a split logging experiment.
    
    Args:
        base_path: Base path without file extension (e.g., "logs/experiment_20240101_120000")
    
    Returns:
        Dictionary with keys 'stats', 'mutations', 'generations', each containing
        a list of log entries
    """
    logs = {
        'stats': [],
        'mutations': [],
        'generations': []
    }
    
    # Load stats
    stats_file = f"{base_path}_stats.jsonl"
    if os.path.exists(stats_file):
        with open(stats_file, 'r', encoding='utf-8') as f:
            for line in f:
                logs['stats'].append(json.loads(line.strip()))
    
    # Load mutations
    mutations_file = f"{base_path}_mutations.jsonl"
    if os.path.exists(mutations_file):
        with open(mutations_file, 'r', encoding='utf-8') as f:
            for line in f:
                logs['mutations'].append(json.loads(line.strip()))
    
    # Load generations
    generations_file = f"{base_path}_generations.jsonl"
    if os.path.exists(generations_file):
        with open(generations_file, 'r', encoding='utf-8') as f:
            for line in f:
                logs['generations'].append(json.loads(line.strip()))
    
    return logs


def analyze_bandit_performance(mutations_log):
    """
    Analyze bandit performance from mutations log.
    
    Args:
        mutations_log: List of mutation log entries
    
    Returns:
        Dictionary with performance metrics per strategy
    """
    strategy_stats = {}
    
    for entry in mutations_log:
        if entry['event'] == 'bandit_update':
            strategy = entry['strategy']
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total': 0,
                    'successes': 0,
                    'failures': 0
                }
            
            strategy_stats[strategy]['total'] += 1
            if entry['is_success']:
                strategy_stats[strategy]['successes'] += 1
            else:
                strategy_stats[strategy]['failures'] += 1
    
    # Calculate success rates
    for strategy in strategy_stats:
        total = strategy_stats[strategy]['total']
        successes = strategy_stats[strategy]['successes']
        strategy_stats[strategy]['success_rate'] = successes / total if total > 0 else 0
    
    return strategy_stats


def analyze_jailbreak_success_rate(generations_log, by_round=False):
    """
    Analyze jailbreak success rate from generations log.
    
    Args:
        generations_log: List of generation log entries
        by_round: If True, return per-round statistics
    
    Returns:
        Overall success rate, or dictionary of per-round rates if by_round=True
    """
    if not by_round:
        total = len(generations_log)
        successes = sum(1 for entry in generations_log if not entry['is_refused'])
        return successes / total if total > 0 else 0
    else:
        round_stats = {}
        for entry in generations_log:
            round_num = entry['round']
            if round_num not in round_stats:
                round_stats[round_num] = {'total': 0, 'successes': 0}
            
            round_stats[round_num]['total'] += 1
            if not entry['is_refused']:
                round_stats[round_num]['successes'] += 1
        
        # Calculate rates
        for round_num in round_stats:
            total = round_stats[round_num]['total']
            successes = round_stats[round_num]['successes']
            round_stats[round_num]['success_rate'] = successes / total if total > 0 else 0
        
        return round_stats


def generate_experiment_report(base_path):
    """
    Generate a comprehensive experiment report from split logs.
    
    Args:
        base_path: Base path to log files
    
    Returns:
        Dictionary with comprehensive experiment analysis
    """
    logs = load_split_logs(base_path)
    
    report = {
        'overview': {},
        'bandit_performance': {},
        'jailbreak_performance': {},
        'timeline': []
    }
    
    # Extract experiment metadata
    if logs['stats'] and logs['stats'][0]['event'] == 'experiment_start':
        report['overview'] = logs['stats'][0]['config']
    
    # Analyze bandit
    report['bandit_performance'] = analyze_bandit_performance(logs['mutations'])
    
    # Analyze jailbreak success
    report['jailbreak_performance'] = {
        'overall': analyze_jailbreak_success_rate(logs['generations']),
        'by_round': analyze_jailbreak_success_rate(logs['generations'], by_round=True)
    }
    
    # Build timeline
    for entry in logs['stats']:
        if entry['event'] in ['round_start', 'round_end']:
            report['timeline'].append({
                'event': entry['event'],
                'round': entry['round'],
                'timestamp': entry['timestamp']
            })
    
    return report
