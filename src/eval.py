import asyncio
from tqdm.asyncio import tqdm
import json

class Evaluator:
    def __init__(self, provider, model_config, benchmark, concurrency=5, reasoning_mode=None):
        self.provider = provider
        self.model_config = model_config
        self.benchmark = benchmark
        self.concurrency = concurrency
        
        # Determine reasoning configuration
        model_has_reasoning = model_config.get('reasoning', False)
        model_reasoning_effort = model_config.get('reasoning_effort')
        
        if reasoning_mode is True:
            if model_has_reasoning:
                self.reasoning_effort = model_reasoning_effort
                self.reasoning_enabled = True
            else:
                print(f"Warning: Model does not support reasoning effort level!")
                self.reasoning_effort = None
                self.reasoning_enabled = True
        elif reasoning_mode is False:
            print(f"Warning: Model does not support reasoning. Continuing without reasoning.")
            self.reasoning_effort = None
            self.reasoning_enabled = False
        else:
            self.reasoning_enabled = model_has_reasoning
            self.reasoning_effort = model_reasoning_effort if model_has_reasoning else None
    
    async def evaluate_single(self, example, semaphore, generation_params):
        """Evaluate a single example with concurrency control"""
        async with semaphore:
            # Prepare prompt
            prompt = self.benchmark.prepare_prompt(example)
            
            # Generate prediction with reasoning configuration
            output = await self.provider.generate(
                model_id=self.model_config['model_id'],
                prompt=prompt,
                params=generation_params,
                reasoning_effort=self.reasoning_effort
            )
            
            # Get reference based on benchmark type
            if hasattr(example, 'reference'):
                target_text = example.reference
            elif hasattr(example, 'answer'):
                target_text = example.answer
            else:
                target_text = None
            
            # Get source text if available (for MT tasks)
            source_text = example.source if hasattr(example, 'source') else None
            
            return {
                'id': example.id,
                'source': source_text,
                'prompt': prompt,
                'reasoning': output['reasoning'],
                'generation': output['generation'],
                'raw_generation': output['raw_generation'],
                'target_text': target_text
            }
    
    async def run(self):
        """Run evaluation on the benchmark asynchronously"""
        # Load benchmark data
        dataset = self.benchmark.load_data()
        
        # Get generation params (task defaults + model overrides)
        generation_params = self.benchmark.get_generation_params(
            self.model_config.get('default_params', {})
        )
        
        # Display reasoning status
        if self.reasoning_enabled:
            if self.reasoning_effort:
                reasoning_status = f"ON (effort={self.reasoning_effort})"
            else:
                reasoning_status = "ON"
        else:
            reasoning_status = "OFF"
        
        print(f"\nEvaluating {self.model_config['name']} on {len(dataset)} examples...")
        print(f"Reasoning: {reasoning_status}")
        print(f"Generation params: {generation_params}")
        print(f"Concurrency: {self.concurrency} requests at a time\n")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # Create tasks for all examples
        tasks = [
            self.evaluate_single(example, semaphore, generation_params)
            for example in dataset
        ]
        
        # Run all tasks with progress bar
        raw_outputs = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
            result = await coro
            raw_outputs.append(result)
        
        # Sort raw_outputs by ID to maintain consistent order
        raw_outputs.sort(key=lambda x: x['id'])
        
        # Extract predictions and references for evaluation
        # Use 'generation' (without reasoning tokens) for evaluation
        predictions = [output['generation'] for output in raw_outputs]
        references = [output['target_text'] for output in raw_outputs]
        
        # Evaluate
        metrics = self.benchmark.evaluate(predictions, references)

        # Add per-example scores to raw outputs if available
        if 'per_example_scores' in metrics.keys():
            per_example_scores = metrics.pop('per_example_scores')
            
            # Add scores to each raw output (now they're aligned by index)
            for i, output in enumerate(raw_outputs):
                output['scores'] = {
                    'xcomet-xl': per_example_scores['xcomet'][i] if 'xcomet' in per_example_scores else None,
                    'bleu': per_example_scores['bleu'][i] if 'bleu' in per_example_scores else None,
                    'chrf': per_example_scores['chrf'][i] if 'chrf' in per_example_scores else None
                }
        
        return {
            'metrics': metrics,
            'raw_outputs': raw_outputs,
            'generation_params': generation_params,
            'reasoning_enabled': self.reasoning_enabled,
            'reasoning_effort': self.reasoning_effort
        }