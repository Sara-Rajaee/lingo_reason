# Linguistic Reasoning

A framework for evaluating language models on different tasks.

## Requirements

- Python 3.x
- API keys for the model providers you intend to use

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sara-Rajaee/lingo_reason
   cd your-repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your API keys as environment variables:
   ```bash
   export TOGETHER_API_KEY=your_key_here
   export GEMINI_API_KEY=your_key_here
   export OPENAI_API_KEY=your_key_here
   ```
   You only need to set the keys for the model providers you plan to use.

## Usage

Run an evaluation by specifying a model and a task:

```bash
python run.py --model gemini-2.5-flash --task polymath
```

To see all available models and tasks:

```bash
python run.py --list
```

## Configuration

Task configurations (e.g. languages, number of evaluation samples) can be modified in the config files located in the `config/` directory.
