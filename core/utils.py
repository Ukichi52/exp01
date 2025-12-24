"""
Utility functions for data loading and logging.
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
    Handles logging of experiment results to JSONL files.
    """
    
    def __init__(self, experiment_name="experiment"):
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            config.LOG_DIR,
            f"{experiment_name}_{timestamp}.jsonl"
        )
        
        # Create log directory if it doesn't exist
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        print(f"[Logger] Logging to {self.log_file}")
    
    def log(self, data):
        """
        Append a single record to the log file.
        
        Args:
            data: Dictionary to log
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def log_round_start(self, round_num):
        """Log the start of a round."""
        self.log({
            'event': 'round_start',
            'round': round_num,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_round_end(self, round_num, stats):
        """Log the end of a round with statistics."""
        self.log({
            'event': 'round_end',
            'round': round_num,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_generation(self, round_num, item_id, prompt, response, 
                       refusal_score, is_refused):
        """Log a generation result."""
        self.log({
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
        """Log a mutation result."""
        self.log({
            'event': 'mutation',
            'round': round_num,
            'item_id': item_id,
            'original_prompt': original_prompt,
            'mutated_prompt': mutated_prompt,
            'strategy': strategy,
            'passed_constraint': passed_constraint,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_error(self, error_msg, context=None):
        """Log an error."""
        self.log({
            'event': 'error',
            'error': error_msg,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })


def format_duration(seconds):
    """Format duration in seconds to human-readable string."""
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
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    
    if iteration == total:
        print()  # New line on completion
        
