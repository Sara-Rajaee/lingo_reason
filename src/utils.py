import yaml
import json
import os
from datetime import datetime
import re

def load_config(config_path):
    """Load YAML configuration file and resolve environment variables"""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace ${VAR} with environment variable values
    def replace_env_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    content = re.sub(r'\$\{(\w+)\}', replace_env_var, content)
    
    return yaml.safe_load(content)

def save_results(results, output_dir, model_name, task_name, subset, reasoning_mode=None):
    """Save evaluation results"""
    # Create directory structure based on reasoning mode
    reasoning_enabled = results.get('reasoning_enabled', False)
    reasoning_effort = results.get('reasoning_effort')
    
    if reasoning_enabled:
        if reasoning_effort:
            model_dir = f"{model_name}_reasoning_{reasoning_effort}"
        else:
            model_dir = f"{model_name}_reasoning"
    else:
        model_dir = f"{model_name}_no_reasoning"
    
    result_path = os.path.join(output_dir, model_dir, task_name, subset)
    os.makedirs(result_path, exist_ok=True)
    
    # Save metadata
    metadata = {
        'model': model_name,
        'task': task_name,
        'subset': subset,
        'reasoning_enabled': reasoning_enabled,
        'reasoning_effort': reasoning_effort,
        'timestamp': datetime.now().isoformat(),
        'generation_params': results.get('generation_params', {})
    }
    with open(os.path.join(result_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save metrics
    with open(os.path.join(result_path, 'metrics.json'), 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    
    # Save raw outputs
    with open(os.path.join(result_path, 'raw_outputs.json'), 'w') as f:
        json.dump(results['raw_outputs'], f, indent=2)
    
    print(f"Results saved to {result_path}")

def get_output_dir():
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join('results', timestamp)