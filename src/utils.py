import yaml
import json
import os
from datetime import datetime
import re

def load_config(config_path):
    """Load YAML configuration file and resolve environment variables"""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace ${VAR} or ${VAR:-default} with environment variable values
    def replace_env_var(match):
        var_name = match.group(1)
        default_value = match.group(2)
        value = os.getenv(var_name)
        if value is not None:
            return value
        if default_value is not None:
            return default_value
        return match.group(0)

    content = re.sub(r'\$\{(\w+)(?::-([^}]*))?\}', replace_env_var, content)
    
    return yaml.safe_load(content)


_RESULT_MARKERS = (
    "metrics.json",
    "raw_outputs.json",
    "sampled_outputs.json",
    "metadata.json",
    "gold_outputs.json",
)


def _path_has_results(path: str) -> bool:
    """True if ``path`` already contains saved eval artifacts."""
    return any(os.path.exists(os.path.join(path, name)) for name in _RESULT_MARKERS)


def allocate_unique_model_run(
    output_dir: str,
    task_name: str,
    model_dir: str,
    subset: str,
    *,
    model_first: bool = False,
) -> tuple:
    """Pick a non-colliding result path for a model×task×subset run.

    Returns ``(result_path, run_index)`` where ``run_index`` is 1 for the first
    run and 2, 3, ... for repeats (directory suffix ``_2``, ``_3``, ...).

    Layout when ``model_first`` is False (default eval results)::
        ``<output>/<task>/<model_dir>[_N]/subset>``

    Layout when ``model_first`` is True (distillation)::
        ``<output>/<model_dir>[_N]/task>/<subset>``
    """
    n = 1
    while True:
        cand_model = model_dir if n == 1 else f"{model_dir}_{n}"
        if model_first:
            result_path = os.path.join(output_dir, cand_model, task_name, subset)
        else:
            result_path = os.path.join(output_dir, task_name, cand_model, subset)
        if not _path_has_results(result_path):
            return result_path, n
        n += 1


def save_results(results, output_dir, model_name, task_name, subset, split, reasoning_mode=None):
    """Save evaluation results"""
    # Create directory structure based on reasoning mode
    reasoning_enabled = results.get('reasoning', False)
    reasoning_effort = results.get('reasoning_effort')
    
    if reasoning_enabled:
        if reasoning_effort:
            model_dir = f"{model_name}_reasoning_{reasoning_effort}"
        else:
            model_dir = f"{model_name}_reasoning"
    else:
        model_dir = f"{model_name}_no_reasoning"

    if task_name == 'polymath':
        task_name = task_name + '_' + split
    
    result_path, run_index = allocate_unique_model_run(
        output_dir, task_name, model_dir, subset
    )
    os.makedirs(result_path, exist_ok=True)
    
    # Save metadata
    metadata = {
        'model': model_name,
        'task': task_name,
        'subset': subset,
        'split': split,
        'reasoning_enabled': reasoning_enabled,
        'reasoning_effort': reasoning_effort,
        'timestamp': datetime.now().isoformat(),
        'generation_params': results.get('generation_params', {}),
        'result_path': result_path,
        'run_index': run_index,
    }
    with open(os.path.join(result_path, 'metadata.json'), 'w', encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Save metrics
    with open(os.path.join(result_path, 'metrics.json'), 'w', encoding="utf-8") as f:
        json.dump(results['metrics'], f, ensure_ascii=False, indent=2)
    
    # Save raw outputs
    with open(os.path.join(result_path, 'raw_outputs.json'), 'w', encoding="utf-8") as f:
        json.dump(results['raw_outputs'], f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {result_path}")
    return result_path

def get_output_dir():
    """Create timestamped output directory"""
    return 'results'

def save_distillation_results(results, output_dir, model_name, task_name, subset, split):
    """Save sampled distillation outputs and the samples that matched the gold answer."""
    if task_name == 'polymath':
        task_name = task_name + '_' + split

    result_path, run_index = allocate_unique_model_run(
        output_dir, task_name, model_name, subset, model_first=True
    )
    os.makedirs(result_path, exist_ok=True)

    metadata = {
        'model': model_name,
        'task': task_name,
        'subset': subset,
        'split': split,
        'reasoning_enabled': results.get('reasoning', False),
        'reasoning_effort': results.get('reasoning_effort'),
        'distillation': results.get('distillation', False),
        'distillation_samples': results.get('distillation_samples', 1),
        'generation_params': results.get('generation_params', {}),
        'num_sampled_outputs': len(results.get('raw_outputs', [])),
        'num_gold_outputs': len(results.get('gold_outputs', [])),
        'result_path': result_path,
        'run_index': run_index,
    }

    with open(os.path.join(result_path, 'metadata.json'), 'w', encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(os.path.join(result_path, 'metrics.json'), 'w', encoding="utf-8") as f:
        json.dump(results['metrics'], f, ensure_ascii=False, indent=2)

    with open(os.path.join(result_path, 'sampled_outputs.json'), 'w', encoding="utf-8") as f:
        json.dump(results['raw_outputs'], f, ensure_ascii=False, indent=2)

    with open(os.path.join(result_path, 'gold_outputs.json'), 'w', encoding="utf-8") as f:
        json.dump(results.get('gold_outputs', []), f, ensure_ascii=False, indent=2)

    with open(os.path.join(result_path, 'all_gold_outputs.json'), 'w', encoding="utf-8") as f:
        json.dump(results['all_gold_outputs'], f, ensure_ascii=False, indent=2)

    print(f"Distillation results saved to {result_path}")
    return result_path


def get_distillation_output_dir():
    """Return output directory for distillation artifacts."""
    return 'distilled_results'
