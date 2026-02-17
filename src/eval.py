import asyncio
from tqdm.asyncio import tqdm
import json

class Evaluator:
    def __init__(self, provider, model_config, benchmark, concurrency=5):
        self.provider = provider
        self.model_config = model_config
        self.benchmark = benchmark
        self.concurrency = concurrency
        
        # Determine reasoning configuration
        model_has_reasoning = model_config['default_params'].get('reasoning', False)
        model_reasoning_effort = model_config['default_params'].get('reasoning_effort', None)
        
        if model_has_reasoning is True:
                self.reasoning_effort = model_reasoning_effort
                self.reasoning = True
        elif model_has_reasoning is False:
            print(f"Warning: Model does not support reasoning. Continuing without reasoning.")
            self.reasoning_effort = None
            self.reasoning = False

    
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

            if hasattr(example, 'source'):
                source_text = example.source
            elif hasattr(example, 'question'):
                source_text = example.question + f"A) {example.A}\nB) {example.B}\nC) {example.C}\nD) {example.D}\n\n"
            else:
                source_text = None
            
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
        if self.reasoning:
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
        def extract_numeric_id(output):
            id_parts = output['id'].rsplit('_', 1)
            return int(id_parts[-1]) if len(id_parts) > 1 else 0

        raw_outputs.sort(key=extract_numeric_id)
        
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
                    'xcomet-xl': per_example_scores['xcomet-xl'][i] if 'xcomet-xl' in per_example_scores else None,
                    'bleu': per_example_scores['bleu'][i] if 'bleu' in per_example_scores else None,
                    'chrfpp': per_example_scores['chrfpp'][i] if 'chrfpp' in per_example_scores else None
                }
        
        return {
            'metrics': metrics,
            'raw_outputs': raw_outputs,
            'generation_params': generation_params,
            'reasoning': self.reasoning,

        }