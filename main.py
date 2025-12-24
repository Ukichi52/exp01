"""
Main Logic Loop for Semantically Constrained Iterative Adversarial Search

Implements the 5-Phase optimization loop:
1. Target Generation (LLaVA with real images)
2. Refusal Scoring (Rule-based)
3. Mutation (LLaVA as LLM optimizer)
4. Semantic Constraint Checking (CLIP)
5. Pruning and Selection (Keep top-K prompts)
"""

import os
import time
import random
from PIL import Image
import torch

import config
from core.model_manager import ModelManager
from core.optimizer import LLaVAOptimizer
from core.scorers.refusal import check_refusal, extract_refusal_reason
from core.scorers.constraint import SemanticConstraint
from core.utils import load_dataset, Logger, format_duration, print_progress_bar


class AdversarialSearchExperiment:
    """
    Main experiment class orchestrating the adversarial search process.
    """
    
    def __init__(self):
        print("="*80)
        print("Semantically Constrained Iterative Adversarial Search")
        print("="*80)
        
        # Initialize components
        self.model_manager = ModelManager(
            llava_path=config.LLAVA_PATH,
            urm_path=config.URM_PATH,
            clip_path=config.CLIP_PATH
        )
        
        self.optimizer = LLaVAOptimizer(self.model_manager)
        self.constraint_checker = SemanticConstraint(self.model_manager)
        self.logger = Logger(experiment_name="adversarial_search")
        
        # Load dataset
        self.dataset = load_dataset(config.DATA_PATH)
        
        # Initialize population with original prompts
        self.population = self._initialize_population()
        
        print(f"\n[Experiment] Initialized with {len(self.population)} items")
        print(f"[Experiment] Max rounds: {config.MAX_ROUNDS}")
        print(f"[Experiment] N_GENERATION: {config.N_GENERATION}")
        print(f"[Experiment] M_MUTATION: {config.M_MUTATION}")
        print(f"[Experiment] TOP_K: {config.TOP_K}")
        print(f"[Experiment] CLIP_THRESHOLD: {config.CLIP_THRESHOLD}")
        print("="*80 + "\n")
    
    def _initialize_population(self):
        """
        Initialize population with original prompts from dataset.
        Each item maintains multiple prompt variants.
        """
        population = []
        
        for item in self.dataset:
            population.append({
                'id': item['id'],
                'image_path': item['image_path'],
                'caption': item['caption'],
                'original_prompt': item['prompt'],
                'prompts': [item['prompt']],  # Start with just the original
                'best_score': 1.0,  # Start with worst score (refusal)
                'best_response': None
            })
        
        return population
    
    def run(self):
        """
        Execute the main optimization loop.
        """
        start_time = time.time()
        
        for round_num in range(1, config.MAX_ROUNDS + 1):
            print(f"\n{'='*80}")
            print(f"ROUND {round_num}/{config.MAX_ROUNDS}")
            print(f"{'='*80}\n")
            
            self.logger.log_round_start(round_num)
            round_start = time.time()
            
            # Phase 1: Target Generation
            generation_results = self.phase_1_target_generation(round_num)
            
            # Phase 2: Refusal Scoring
            scored_results = self.phase_2_scoring(round_num, generation_results)
            
            # Phase 3 & 4: Mutation and Constraint Checking
            mutations = self.phase_3_4_mutation_and_constraint(
                round_num, scored_results
            )
            
            # Phase 5: Pruning and Selection
            self.phase_5_pruning_and_selection(round_num, mutations)
            
            # Round statistics
            round_duration = time.time() - round_start
            stats = self._compute_round_stats(scored_results)
            
            print(f"\n[Round {round_num}] Statistics:")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Average score: {stats['avg_score']:.3f}")
            print(f"  Duration: {format_duration(round_duration)}")
            
            self.logger.log_round_end(round_num, stats)
        
        total_duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Experiment completed in {format_duration(total_duration)}")
        print(f"{'='*80}\n")
    
    def phase_1_target_generation(self, round_num):
        """
        Phase 1: Generate responses using LLaVA with real images.
        Generate N responses per prompt variant.
        """
        print(f"[Phase 1] Target Generation (N={config.N_GENERATION})")
        
        # Load LLaVA for target generation
        model, processor = self.model_manager.load_llava()
        
        generation_results = []
        total_generations = sum(len(item['prompts']) for item in self.population)
        progress = 0
        
        for item in self.population:
            image = Image.open(item['image_path']).convert('RGB')
            
            for prompt in item['prompts']:
                # Generate N responses for this prompt
                for gen_idx in range(config.N_GENERATION):
                    response = self._generate_response(
                        model, processor, image, prompt
                    )
                    
                    generation_results.append({
                        'item_id': item['id'],
                        'prompt': prompt,
                        'response': response,
                        'original_prompt': item['original_prompt']
                    })
                    
                    progress += 1
                    print_progress_bar(
                        progress, 
                        total_generations * config.N_GENERATION,
                        prefix='  Generating',
                        suffix=f'{progress}/{total_generations * config.N_GENERATION}'
                    )
        
        print(f"  Generated {len(generation_results)} responses\n")
        return generation_results
    
    def phase_2_scoring(self, round_num, generation_results):
        """
        Phase 2: Score responses using rule-based refusal detection.
        """
        print(f"[Phase 2] Refusal Scoring")
        
        scored_results = []
        
        for result in generation_results:
            is_refused, score = check_refusal(result['response'])
            
            scored_result = {
                **result,
                'refusal_score': score,
                'is_refused': is_refused
            }
            
            scored_results.append(scored_result)
            
            # Log the generation
            self.logger.log_generation(
                round_num=round_num,
                item_id=result['item_id'],
                prompt=result['prompt'],
                response=result['response'],
                refusal_score=score,
                is_refused=is_refused
            )
        
        success_count = sum(1 for r in scored_results if not r['is_refused'])
        print(f"  Successful: {success_count}/{len(scored_results)} "
              f"({100*success_count/len(scored_results):.1f}%)\n")
        
        return scored_results
    
    def phase_3_4_mutation_and_constraint(self, round_num, scored_results):
        """
        Phase 3 & 4: Generate mutations for failed prompts and check constraints.
        """
        print(f"[Phase 3-4] Mutation & Semantic Constraint")
        
        # Identify failed prompts that need mutation
        failed_items = {}
        for result in scored_results:
            if result['is_refused']:
                item_id = result['item_id']
                if item_id not in failed_items:
                    failed_items[item_id] = {
                        'prompt': result['prompt'],
                        'original_prompt': result['original_prompt'],
                        'response': result['response']
                    }
        
        if not failed_items:
            print("  No failed prompts to mutate\n")
            return []
        
        print(f"  Mutating {len(failed_items)} failed prompts...")
        
        all_mutations = []
        
        for item_id, failed_data in failed_items.items():
            prompt = failed_data['prompt']
            original = failed_data['original_prompt']
            
            # Generate M mutations using different strategies
            mutations_for_item = []
            
            for mut_idx in range(config.M_MUTATION):
                # Select random strategy
                strategy = random.choice(config.ATTACK_STRATEGIES)
                
                # Extract refusal reason for targeted mutation
                refusal_reason = extract_refusal_reason(failed_data['response'])
                
                # Generate mutation with retry logic for semantic constraint
                mutation = None
                for retry in range(config.MAX_SEMANTIC_RETRIES):
                    candidate = self.optimizer.mutate(
                        prompt, strategy, refusal_reason
                    )
                    
                    # Check semantic constraint
                    is_valid, similarity = self.constraint_checker.check_similarity_single(
                        candidate, original
                    )
                    
                    if is_valid:
                        mutation = candidate
                        break
                    else:
                        print(f"    Retry {retry+1}/{config.MAX_SEMANTIC_RETRIES} "
                              f"(similarity: {similarity:.3f})")
                
                if mutation:
                    mutations_for_item.append(mutation)
                    
                    # Log successful mutation
                    self.logger.log_mutation(
                        round_num=round_num,
                        item_id=item_id,
                        original_prompt=original,
                        mutated_prompt=mutation,
                        strategy=strategy,
                        passed_constraint=True
                    )
                else:
                    print(f"    [Warning] Failed to generate valid mutation "
                          f"for {item_id} after {config.MAX_SEMANTIC_RETRIES} retries")
            
            if mutations_for_item:
                all_mutations.append({
                    'item_id': item_id,
                    'mutations': mutations_for_item
                })
        
        total_mutations = sum(len(m['mutations']) for m in all_mutations)
        print(f"  Generated {total_mutations} valid mutations\n")
        
        return all_mutations
    
    def phase_5_pruning_and_selection(self, round_num, mutations):
        """
        Phase 5: Test mutations and select top-K prompts for next round.
        """
        print(f"[Phase 5] Pruning & Selection (TOP_K={config.TOP_K})")
        
        if not mutations:
            print("  No mutations to evaluate\n")
            return
        
        # Load LLaVA for testing mutations
        model, processor = self.model_manager.load_llava()
        
        # Evaluate each mutation
        for mutation_batch in mutations:
            item_id = mutation_batch['item_id']
            
            # Find the corresponding item in population
            item = next(i for i in self.population if i['id'] == item_id)
            image = Image.open(item['image_path']).convert('RGB')
            
            candidates = []
            
            for mutation in mutation_batch['mutations']:
                # Generate response for mutation
                response = self._generate_response(
                    model, processor, image, mutation
                )
                
                # Score the response
                is_refused, score = check_refusal(response)
                
                candidates.append({
                    'prompt': mutation,
                    'response': response,
                    'score': score,
                    'is_refused': is_refused
                })
            
            # Add existing prompts to candidates
            for existing_prompt in item['prompts']:
                # Use cached best response if available
                candidates.append({
                    'prompt': existing_prompt,
                    'response': item.get('best_response', ''),
                    'score': item['best_score'],
                    'is_refused': item['best_score'] > 0.5
                })
            
            # Sort by score (ascending - lower is better)
            candidates.sort(key=lambda x: x['score'])
            
            # Select top-K
            top_k = candidates[:config.TOP_K]
            
            # Update population
            item['prompts'] = [c['prompt'] for c in top_k]
            item['best_score'] = top_k[0]['score']
            item['best_response'] = top_k[0]['response']
            
            print(f"  Item {item_id}: Selected {len(top_k)} prompts "
                  f"(best score: {top_k[0]['score']:.3f})")
        
        print()
    
    def _generate_response(self, model, processor, image, prompt):
        """
        Generate a single response from LLaVA.
        """
        # LLaVA expects a conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text_prompt = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                do_sample=True
            )
        
        # Decode only the generated part (exclude input prompt)
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response = processor.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def _compute_round_stats(self, scored_results):
        """
        Compute statistics for a round.
        """
        total = len(scored_results)
        success_count = sum(1 for r in scored_results if not r['is_refused'])
        avg_score = sum(r['refusal_score'] for r in scored_results) / total
        
        return {
            'total': total,
            'success_count': success_count,
            'success_rate': success_count / total if total > 0 else 0,
            'avg_score': avg_score
        }


def main():
    """
    Entry point for the experiment.
    """
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Create experiment instance
    experiment = AdversarialSearchExperiment()
    
    # Run the experiment
    try:
        experiment.run()
    except KeyboardInterrupt:
        print("\n\n[Experiment] Interrupted by user")
    except Exception as e:
        print(f"\n\n[Experiment] Error: {e}")
        experiment.logger.log_error(str(e))
        raise
    finally:
        # Cleanup
        experiment.model_manager.unload_model()
        print("[Experiment] Cleanup complete")


if __name__ == "__main__":
    main()
    
