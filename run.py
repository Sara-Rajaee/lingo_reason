#!/usr/bin/env python3
import asyncio
import argparse
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from src.api import ProviderFactory
from src.tasks import BenchmarkFactory
from src.eval import Evaluator
from src.utils import load_config, save_results, get_output_dir

async def evaluate_task_subset(model, provider_config, task_name, task_config, subset, concurrency=1):
    """Evaluate a single model on a task subset"""
    # Initialize provider
    provider = ProviderFactory.get_provider(model['provider'], provider_config)
    
    # Initialize benchmark for this subset
    benchmark = BenchmarkFactory.get_benchmark(
        task_config['benchmark_type'], 
        task_config,
        subset
    )
    # Run evaluation
    evaluator = Evaluator(provider, model, benchmark, concurrency=5)
    results = await evaluator.run()
    
    # Save results
    output_dir = get_output_dir()
    save_results(results, output_dir, model['name'], task_name, subset)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Model: {model['name']}")
    print(f"Task: {task_name}")
    print(f"Subset: {subset}")
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")
    print(f"{'='*60}\n")
    
    return results

async def evaluate_task(model, provider_config, task_name, task_config, concurrency=5):
    """Evaluate a model on all subsets of a task"""
    subsets = task_config.get('subsets', [])
    
    if not subsets:
        print(f"No subsets defined for task: {task_name}")
        return []
    
    print(f"\n{'#'*60}")
    print(f"Starting evaluation: {model['name']} on {task_name}")
    print(f"Subsets: {', '.join(subsets)}")
    print(f"{'#'*60}\n")
    
    all_results = []
    
    # Evaluate each subset sequentially
    for subset in subsets:
        result = await evaluate_task_subset(
            model, 
            provider_config, 
            task_name, 
            task_config, 
            subset,
            concurrency
        )
        all_results.append({
            'subset': subset,
            'metrics': result['metrics']
        })
    
    return all_results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate LLM models on benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a specific model on a specific task
  python run.py --model deepseek-r1 --task mmmlu
  
  # Evaluate all models on a task
  python run.py --task mmmlu
  
  # Evaluate a model on all tasks
  python run.py --model deepseek-r1
  
  # List available models and tasks
  python run.py --list
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        help='Name of the model to evaluate (as defined in config/models.yaml)'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        help='Name of the task to evaluate (as defined in config/benchmarks.yaml)'
    )
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List available models and tasks'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        help='Override concurrency/batch_size (default: use task config)'
    )
    
    return parser.parse_args()

def list_available_options(models_config, tasks_config):
    """List available models and tasks"""
    print("\n" + "="*60)
    print("AVAILABLE MODELS:")
    print("="*60)
    for model in models_config['models']:
        print(f"  - {model['name']:20} (provider: {model['provider']})")
    
    print("\n" + "="*60)
    print("AVAILABLE TASKS:")
    print("="*60)
    for task_name, task_config in tasks_config['tasks'].items():
        subsets = task_config.get('subsets', [])
        print(f"  - {task_name:20} ({len(subsets)} subsets: {', '.join(subsets[:3])}{'...' if len(subsets) > 3 else ''})")
    print()

async def main():
    args = parse_args()
    
    # Load configurations
    providers_config = load_config('config/providers.yaml')
    models_config = load_config('config/models.yaml')
    tasks_config = load_config('config/tasks.yaml')
    
    # List mode
    if args.list:
        list_available_options(models_config, tasks_config)
        return
    
    # Determine which models to evaluate
    if args.model:
        # Find specific model
        models_to_eval = [m for m in models_config['models'] if m['name'] == args.model]
        if not models_to_eval:
            print(f"Error: Model '{args.model}' not found in config/models.yaml")
            print("\nAvailable models:")
            for model in models_config['models']:
                print(f"  - {model['name']}")
            return
    else:
        # Evaluate all models
        models_to_eval = models_config['models']
    
    # Determine which tasks to evaluate
    if args.task:
        # Find specific task
        if args.task not in tasks_config['tasks']:
            print(f"Error: Task '{args.task}' not found in config/benchmarks.yaml")
            print("\nAvailable tasks:")
            for task_name in tasks_config['tasks'].keys():
                print(f"  - {task_name}")
            return
        tasks_to_eval = {args.task: tasks_config['tasks'][args.task]}
    else:
        # Evaluate all tasks
        tasks_to_eval = tasks_config['tasks']
    
    # Run evaluations
    for model in models_to_eval:
        provider_config = providers_config[model['provider']]
        
        for task_name, task_config in tasks_to_eval.items():
            # Determine concurrency
            if args.concurrency:
                concurrency = args.concurrency
            else:
                concurrency = task_config['defaults'].get('batch_size', 5) 

            # Run evaluation on all subsets
            results = await evaluate_task(
                model, 
                provider_config, 
                task_name, 
                task_config,
                concurrency
            )
            
            # Print aggregate summary
            print(f"\n{'='*60}")
            print(f"AGGREGATE RESULTS: {model['name']} on {task_name}")
            print(f"{'='*60}")
            for result in results:
                print(f"{result['subset']:20} -> {result['metrics']}")

if __name__ == "__main__":
    asyncio.run(main())