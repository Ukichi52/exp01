"""
Semantic Constraint Checker using CLIP
Ensures mutated prompts maintain semantic similarity to original.
"""

import torch
import torch.nn.functional as F
import config


class SemanticConstraint:
    """
    Uses CLIP to verify semantic similarity between original and mutated prompts.
    Implements batch processing for efficiency.
    """
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.threshold = config.CLIP_THRESHOLD
    
    def check_similarity(self, text_list, original_text):
        """
        Check semantic similarity between candidates and original text.
        
        Args:
            text_list: List of candidate text strings to check
            original_text: The original reference text
        
        Returns:
            valid_indices: List of indices where similarity > threshold
        """
        if not text_list:
            return []
        
        # Load CLIP model
        model, processor = self.model_manager.load_clip()
        
        # Encode original text
        original_inputs = processor(
            text=[original_text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            original_features = model.get_text_features(**original_inputs)
            original_features = F.normalize(original_features, p=2, dim=-1)
        
        # Batch encode all candidate texts
        candidate_inputs = processor(
            text=text_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            candidate_features = model.get_text_features(**candidate_inputs)
            candidate_features = F.normalize(candidate_features, p=2, dim=-1)
        
        # Compute cosine similarities (batch operation)
        similarities = torch.mm(candidate_features, original_features.T).squeeze(-1)
        
        # Find valid indices where similarity exceeds threshold
        valid_mask = similarities > self.threshold
        valid_indices = torch.where(valid_mask)[0].cpu().tolist()
        
        # Log similarity scores for debugging
        print(f"[SemanticConstraint] Checked {len(text_list)} candidates")
        print(f"[SemanticConstraint] Valid candidates: {len(valid_indices)}/{len(text_list)}")
        
        for idx, sim in enumerate(similarities.cpu().tolist()):
            status = "✓" if idx in valid_indices else "✗"
            print(f"  {status} Candidate {idx}: similarity = {sim:.3f}")
        
        return valid_indices
    
    def check_similarity_single(self, candidate_text, original_text):
        """
        Check similarity for a single candidate (convenience method).
        
        Args:
            candidate_text: Single candidate text string
            original_text: The original reference text
        
        Returns:
            Tuple of (is_valid: bool, similarity: float)
        """
        valid_indices = self.check_similarity([candidate_text], original_text)
        
        # If index 0 is in valid_indices, the candidate passed
        is_valid = 0 in valid_indices
        
        # Recompute similarity for return value
        model, processor = self.model_manager.load_clip()
        
        texts = [original_text, candidate_text]
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = F.normalize(features, p=2, dim=-1)
            similarity = torch.mm(features[1:2], features[0:1].T).item()
        
        return is_valid, similarity