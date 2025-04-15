# Template-oriented Prompting: A New Paradigm for Enhanced LLM Reasoning

## Overview

This project introduces Template-oriented Prompting (ToP), a novel paradigm designed to enhance reasoning capabilities in Large Language Models (LLMs). ToP leverages structured templates to guide LLMs through complex reasoning tasks, resulting in improved performance across various benchmarks.

## Key Features

- **Meta-Buffer System**: Implements a buffer mechanism that stores and retrieves templates for different reasoning tasks
- **Template Distillation**: Automatically extracts and refines reasoning patterns from successful LLM responses
- **Multi-Task Support**: Currently supports multiple reasoning benchmarks including:
  - Game of 24 (arithmetic reasoning)
  - Checkmate in One (chess reasoning)
  - Word Sorting (alphabetical ordering)

## Results

Our Template-oriented Prompting (ToP) approach demonstrates significant improvements over existing methods on the Game of 24 benchmark:

| Method | Accuracy (%) |
|--------|-------------|
| Standard Prompting | 3.0 |
| Zero-shot CoT | 11.0 |
| Expert Prompting | 3.0 |
| PAL | 64.0 |
| Tree-of-Thought | 74.0 |
| Graph-of-Thought | 73.2 |
| Meta Prompting | 67.0 |
| Buffer-of-Thoughts | 82.4 |
| **Template-oriented Prompting (ToP)** | **90.0** |

ToP achieves a **90.0%** accuracy on the Game of 24 benchmark, representing an **7.6%** improvement over the previous state-of-the-art method.


## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_benchmarks.py --task_name gameof24 --api_key <your_api_key> --model_id gpt-4o
```

## Acknowledgments

This project was produced by Zochi, an Artificial Scientist from Intology.


