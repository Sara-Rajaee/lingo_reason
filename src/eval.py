import asyncio
from tqdm.asyncio import tqdm
import json
from dataclasses import asdict, is_dataclass

class Evaluator:
    def __init__(
        self,
        provider,
        model_config,
        benchmark,
        task_config,
        concurrency=5,
        distillation=False,
        distillation_samples=1,
    ):
        self.provider = provider
        self.model_config = model_config
        self.benchmark = benchmark
        self.concurrency = concurrency
        self.task_config = task_config
        self.distillation = distillation
        self.distillation_samples = max(1, distillation_samples)

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

    
    async def evaluate_single(self, example, semaphore, generation_params, sample_index=None):
        """Evaluate a single example with concurrency control"""
        async with semaphore:
            # Prepare prompt
            prompt = self.benchmark.prepare_prompt(example)
            system_prompt = self.benchmark.prepare_system_prompt()
            # Generate prediction with reasoning configuration
            output = await self.provider.generate(
                model_id=self.model_config['model_id'],
                prompt=prompt,
                params=generation_params,
                system_prompt=system_prompt,
                reasoning_effort=self.reasoning_effort,
                thinking_budget=self.model_config['default_params'].get('thinking_budget', 0) # setting to 0 for disabling reasoning 
            )
            
            # Get reference based on benchmark type
            if hasattr(example, 'reference'):
                target_text = example.reference
            elif hasattr(example, 'answer'):
                target_text = example.answer
            elif hasattr(example, 'original_context'):  # AbsenceBench
                target_text = example.omitted_context 
            else:
                target_text = None
            
            # Get source text if available (for MT tasks)
            source_text = example.source if hasattr(example, 'source') else None

            if hasattr(example, 'source'): #mt
                source_text = example.source
                eval_type = 'mt_metrics'
                points = 1
            elif hasattr(example, 'question') and hasattr(example, 'A'): #mmlu
                source_text = example.question + f"A) {example.A}\nB) {example.B}\nC) {example.C}\nD) {example.D}\n\n"
                eval_type = 'accuracy'
                points = 1
            elif hasattr(example, 'question'): #polymath
                source_text = example.question 
                eval_type = 'accuracy'
                points = 1
            elif hasattr(example, 'eval_type') and hasattr(example, 'points'): #MuLR
                eval_type = example.eval_type
                source_text = example.prompt
                points = example.points
            elif hasattr(example, 'original_context'):
                source_text = example.original_context
                eval_type = 'f1'
                points = 1
            else:
                source_text = None
                eval_type = None
                points = 1
            raw_output = {

                'id': example.id,
                '_task_fields': self._get_task_fields(example),
                'source': source_text,
                'prompt': prompt,
                'reasoning': output['reasoning'],
                'generation': output['generation'],
                'raw_generation': output['raw_generation'],
                'finish_reason': output.get('finish_reason'),
                'target_text': target_text,
                'eval_type': eval_type,
                'points': points
            }
            if sample_index is not None:
                raw_output['sample_index'] = sample_index
                raw_output['sample_id'] = f"{example.id}_sample_{sample_index}"
            return raw_output

    def _get_task_fields(self, example):
        """Return the original task/example columns for distilled outputs."""
        if is_dataclass(example):
            return asdict(example)
        return {
            key: value
            for key, value in vars(example).items()
            if not key.startswith('_')
        }

    def _has_gold_answer(self, output):
        """Infer whether a sampled output matched the gold answer from per-example scores."""
        scores = output.get('scores', {})
        if scores.get('accuracy') == 1:
            return True
        if scores.get('f1') == 1:
            return True
        if scores.get('points') is not None:
            return scores['points'] >= output.get('points', 1)
        if output.get('target_text') is not None:
            return output.get('generation') == output.get('target_text')
        return False
    
    def _build_distilled_output(self, output):
        """Keep only the original task/example columns plus the sampled reasoning trace."""
        distilled_output = output.get('_task_fields', {}).copy()
        distilled_output['reasoning'] = output.get('reasoning')
        distilled_output['prompt'] = output.get('prompt')
        return distilled_output

    async def run(self):
        """Run evaluation on the benchmark asynchronously"""
        # Load benchmark data
        dataset = self.benchmark.load_data()
        
        # Get generation params (task defaults + model overrides)
        generation_params = self.benchmark.get_generation_params(
            self.model_config.get('default_params', {})
        )
        if self.distillation:
            task_defaults = self.task_config.get('defaults', {})
            distillation_temperature = task_defaults.get(
                'distillation_temperature',
                task_defaults.get('temperature')
            )
            if distillation_temperature is not None:
                generation_params['temperature'] = distillation_temperature
            distillation_top_p = task_defaults.get(
                'distillation_top_p',
                task_defaults.get('top_p')
            )
            if distillation_top_p is not None:
                generation_params['top_p'] = distillation_top_p
        
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
        if self.distillation:
            print(f"Distillation: ON ({self.distillation_samples} samples per example)\n")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        # Create tasks for all examples
        tasks = []
        for example in dataset:
            if self.distillation:
                tasks.extend(
                    self.evaluate_single(example, semaphore, generation_params, sample_index)
                    for sample_index in range(self.distillation_samples)
                )
            else:
                tasks.append(self.evaluate_single(example, semaphore, generation_params))
     
        # Run all tasks with progress bar
        raw_outputs = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
            result = await coro
            raw_outputs.append(result)
        
        # Sort raw_outputs by ID to maintain consistent order
        def extract_numeric_id(output):
            id_parts = output['id'].rsplit('_', 1)
            try:
                example_id = int(id_parts[-1]) if len(id_parts) > 1 else 0
                sort_key = (0, example_id)
            except ValueError:
                sort_key = (1, output['id'])
            return (*sort_key, output.get('sample_index', 0))

        raw_outputs.sort(key=extract_numeric_id)
        
        # Extract predictions and references for evaluation
        # Use 'generation' (without reasoning tokens) for evaluation
        # Points can be used to weigh examples
        # Eval types are relevant for linguistic reasoning with diverse tasks types
        predictions = [output['generation'] for output in raw_outputs]
        references = [output['target_text'] for output in raw_outputs]
        eval_types = [output['eval_type'] for output in raw_outputs]
        points = [output['points'] for output in raw_outputs]
        
        # Evaluate
        metrics = self.benchmark.evaluate(predictions, references, eval_types, points)

        # Add per-example scores to raw outputs if available
        if 'per_example_scores' in metrics.keys():
            per_example_scores = metrics.pop('per_example_scores')
            
            # Add scores to each raw output (now they're aligned by index)
            for i, output in enumerate(raw_outputs):
                output['scores'] = {
                    'bleu': per_example_scores['bleu'][i] if 'bleu' in per_example_scores else None,
                    'chrfpp': per_example_scores['chrfpp'][i] if 'chrfpp' in per_example_scores else None
                }
                if self.task_config['defaults'].get('include_comet', False):
                     output['scores'].update({
                    'xcomet-xl': per_example_scores['xcomet-xl'][i] if 'xcomet-xl' in per_example_scores else None})

        elif 'per_example_accuracy' in metrics.keys():
            per_example_scores = metrics.pop('per_example_accuracy')
            # Add scores to each raw output (now they're aligned by index)
            for i, output in enumerate(raw_outputs):
                output['scores'] = {
                    'accuracy': per_example_scores[i]
                }

        elif 'per_example_stats' in metrics.keys(): # MuLR
            per_example_stats = metrics.pop('per_example_stats')
            # Add scores to each raw output (now they're aligned by index)
            for i, output in enumerate(raw_outputs):
                output['scores'] = {
                    'points': per_example_stats['points'][i],
                    'valid_format': per_example_stats['valid_formats'][i],
                    'extracted_answer': per_example_stats['extracted_answers'][i]
                }
        elif 'per_example_f1' in metrics.keys(): #AbsenceBench
            per_example_f1 = metrics.pop('per_example_f1')
            # Add scores to each raw output (now they're aligned by index)
            for i, output in enumerate(raw_outputs):
                output['scores'] = {
                    'f1': per_example_f1[i],
                }
        if self.distillation:
            gold_outputs = []
            correct_sampled_output = []
            seen_example_ids = set()

            for output in raw_outputs:
                output['has_gold_answer'] = self._has_gold_answer(output)
                if output['has_gold_answer']:
                    correct_sampled_output.append(self._build_distilled_output(output))
                    if output['id'] not in seen_example_ids:
                        gold_outputs.append(self._build_distilled_output(output))
                        seen_example_ids.add(output['id'])
        else:
            gold_outputs = []
        for output in raw_outputs:
            output.pop('_task_fields', None)
        return {
            'metrics': metrics,
            'raw_outputs': raw_outputs,
            'gold_outputs': gold_outputs,
            'all_gold_outputs': correct_sampled_output,
            'generation_params': generation_params,
            'reasoning': self.reasoning,
            'reasoning_effort': self.reasoning_effort,
            'distillation': self.distillation,
            'distillation_samples': self.distillation_samples,
        }