"""
Evaluation script for Template-oriented Prompting (ToP) framework on Game of 24.

This script evaluates and compares the performance of the ToP method with the baseline BFS
method on the Game of 24 benchmark. It measures success rates, solving times, and other
relevant metrics to assess the effectiveness of the template-oriented approach.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import random
import re

# Import required modules
from game24_solver import Game24Solver
from api_client import ApiClient
from logger import ToP_Logger

# Set up logging
logger = ToP_Logger(name="top_evaluation", log_dir="evaluation_logs")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate ToP framework on Game of 24')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--model_id', type=str, default='gpt-4-turbo', 
                        help='Model to use for evaluation')
    parser.add_argument('--baseline_model_id', type=str, default='gpt-4-turbo', 
                        help='Model to use for baseline evaluation')
    parser.add_argument('--benchmark_path', type=str, 
                        default='/Users/andyzhou/Documents/zochi/zochi/workspace/top/benchmarks/gameof24.jsonl',
                        help='Path to benchmark data')
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--sample_size', type=int, default=50,
                        help='Number of examples to sample from benchmark (0 for all)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--baseline_only', action='store_true',
                        help='Only evaluate the baseline method')
    parser.add_argument('--top_only', action='store_true',
                        help='Only evaluate the ToP method')
    
    return parser.parse_args()


def load_benchmark_data(benchmark_path: str, sample_size: int = 0, random_seed: int = 42) -> List[List[int]]:
    """
    Load benchmark data from the specified path.
    
    Args:
        benchmark_path: Path to the benchmark JSONL file
        sample_size: Number of examples to sample (0 for all)
        random_seed: Random seed for sampling
        
    Returns:
        List of examples, where each example is a list of four integers
    """
    logger.info(f"Loading benchmark data from {benchmark_path}")
    
    examples = []
    
    try:
        with open(benchmark_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                input_str = data.get('input', '').strip()
                
                # Extract numbers from input string
                numbers = [int(n) for n in re.findall(r'\d+', input_str)]
                
                # Ensure we have exactly 4 numbers
                if len(numbers) == 4:
                    examples.append(numbers)
                else:
                    logger.warning(f"Skipping invalid input: {input_str} (found {len(numbers)} numbers)")
    except Exception as e:
        logger.error(f"Error loading benchmark data: {e}")
        raise
    
    logger.info(f"Loaded {len(examples)} examples from benchmark")
    
    # Sample if requested
    if sample_size > 0 and sample_size < len(examples):
        random.seed(random_seed)
        examples = random.sample(examples, sample_size)
        logger.info(f"Sampled {sample_size} examples for evaluation")
    
    return examples


def verify_solution(numbers: List[int], solution: str) -> bool:
    """
    Verify that a solution is correct for the Game of 24.
    
    Args:
        numbers: List of four integers for the Game of 24 problem
        solution: Solution expression to verify
        
    Returns:
        Boolean indicating whether the solution is valid
    """
    # Handle None or empty solution
    if not solution:
        logger.warning(f"Empty solution for numbers {numbers}")
        return False
        
    # Log the raw solution for debugging
    logger.debug(f"Verifying solution: '{solution}' for numbers {numbers}")
    
    # Extract expression if it contains an equals sign or explanation text
    if "=" in solution:
        expression = solution.split("=")[0].strip()
        logger.debug(f"Extracted expression before '=': '{expression}'")
    elif "solution:" in solution.lower():
        expression = solution.split("solution:", 1)[1].strip()
        logger.debug(f"Extracted expression after 'solution:': '{expression}'")
    else:
        expression = solution.strip()
    
    # Clean up the expression
    original_expression = expression
    expression = expression.replace('×', '*').replace('÷', '/')
    if original_expression != expression:
        logger.debug(f"Cleaned expression: '{expression}'")
    
    # Extract all numbers from the expression
    all_nums_in_expr = [int(n) for n in re.findall(r'\d+', expression)]
    logger.debug(f"Numbers found in expression: {all_nums_in_expr}")
    
    # Check if all numbers from the input are used exactly once
    number_counts = {}
    for num in numbers:
        number_counts[num] = number_counts.get(num, 0) + 1
        
    expr_number_counts = {}
    for num in all_nums_in_expr:
        expr_number_counts[num] = expr_number_counts.get(num, 0) + 1
    
    numbers_correct = True
    for num in numbers:
        if expr_number_counts.get(num, 0) != number_counts.get(num, 0):
            numbers_correct = False
            logger.debug(f"Number mismatch: {num} appears {expr_number_counts.get(num, 0)} times instead of {number_counts.get(num, 0)}")
            break
            
    # Check if the expression evaluates to 24
    try:
        result = eval(expression)
        value_correct = abs(result - 24) < 1e-6
        logger.debug(f"Expression '{expression}' evaluates to {result}, value_correct={value_correct}")
    except Exception as e:
        logger.warning(f"Error evaluating expression '{expression}': {e}")
        value_correct = False
        
    is_valid = numbers_correct and value_correct
    if not is_valid:
        logger.debug(f"Invalid solution: '{solution}' for numbers {numbers} " +
                    f"(numbers_correct: {numbers_correct}, value_correct: {value_correct})")
    else:
        logger.debug(f"Valid solution confirmed: '{solution}' for numbers {numbers}")
    
    return is_valid


def evaluate_baseline_bfs(
    examples: List[List[int]], 
    api_key: Optional[str] = None,
    model_id: str = "gpt-4-turbo"
) -> Dict[str, Any]:
    """
    Evaluate the baseline BFS method on the benchmark examples.
    
    Args:
        examples: List of benchmark examples to evaluate
        api_key: OpenAI API key
        model_id: Model ID to use
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating baseline BFS method on {len(examples)} examples")
    
    # BFS template for Game of 24
    bfs_template = """
from itertools import permutations, product

def find_solution(numbers):
    # Find a solution to the Game of 24 using BFS over all possible expressions.
    ops = '+-*/'
    for nums in permutations(numbers):
        for op_combination in product(ops, repeat=3):
            expressions = [
                f"(({nums[0]} {op_combination[0]} {nums[1]}) {op_combination[1]} {nums[2]}) {op_combination[2]} {nums[3]}",
                f"({nums[0]} {op_combination[0]} ({nums[1]} {op_combination[1]} {nums[2]})) {op_combination[2]} {nums[3]}",
                f"{nums[0]} {op_combination[0]} (({nums[1]} {op_combination[1]} {nums[2]}) {op_combination[2]} {nums[3]})",
                f"{nums[0]} {op_combination[0]} ({nums[1]} {op_combination[1]} ({nums[2]} {op_combination[2]} {nums[3]}))",
                f"({nums[0]} {op_combination[0]} {nums[1]}) {op_combination[1]} ({nums[2]} {op_combination[2]} {nums[3]})"
            ]

            for expr in expressions:
                try:
                    if abs(eval(expr) - 24) < 1e-6:
                        return expr
                except (ZeroDivisionError, SyntaxError):
                    continue
    return "No solution found."

# Example usage
numbers = NUMBERS
solution = find_solution(numbers)
print(solution)
"""
    
    # Initialize API client
    api_client = ApiClient(
        api_key=api_key,
        base_url="https://api.openai.com/v1/",
        model=model_id
    )
    
    # Track results
    results = []
    successful = 0
    total_time = 0
    
    # Evaluate each example
    for i, numbers in enumerate(examples):
        start_time = time.time()
        
        try:
            logger.info(f"Evaluating baseline BFS method on example {i+1}/{len(examples)}: {numbers}")
            
            # Create prompt for this example
            template = bfs_template.replace("NUMBERS", str(numbers))
            prompt = f"""
You are a Python expert solving the Game of 24. Given four numbers, find an arithmetic expression 
that equals 24 using each number exactly once and only +, -, *, / operations and parentheses.

Please implement the solution for numbers {numbers} using the following Python code template.
Return ONLY the output of running this code without any additional text.

```python
{template}
```
"""
            
            # Generate solution
            with logger.create_timed_section(f"baseline_bfs_example_{i}"):
                solution = api_client.generate(prompt, temperature=0.2)
            
            # Verify solution
            is_correct = verify_solution(numbers, solution)
            
            # Calculate time
            solving_time = time.time() - start_time
            
            # Record result
            result = {
                "numbers": numbers,
                "method": "baseline_bfs",
                "solution": solution,
                "is_correct": is_correct,
                "solving_time": solving_time
            }
            results.append(result)
            
            if is_correct:
                successful += 1
                
            total_time += solving_time
            
            logger.info(f"Baseline BFS example {i+1}: " +
                     f"correct={is_correct}, time={solving_time:.2f}s, " +
                     f"solution={solution[:50]}...")
            
        except Exception as e:
            logger.error(f"Error evaluating baseline BFS on example {i+1}: {e}")
            results.append({
                "numbers": numbers,
                "method": "baseline_bfs",
                "error": str(e),
                "is_correct": False,
                "solving_time": time.time() - start_time
            })
    
    # Calculate metrics
    success_rate = successful / len(examples) if examples else 0
    avg_time = total_time / len(examples) if examples else 0
    
    logger.info(f"Baseline BFS evaluation completed: " +
             f"{successful}/{len(examples)} successful ({success_rate:.2%}), " +
             f"avg time: {avg_time:.2f}s")
    
    return {
        "method": "baseline_bfs",
        "total_examples": len(examples),
        "successful": successful,
        "success_rate": success_rate,
        "average_solving_time": avg_time,
        "results": results
    }


async def evaluate_top_method(
    examples: List[List[int]], 
    api_key: Optional[str] = None,
    model_id: str = "gpt-4-turbo",
    results_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    Evaluate the ToP method on the benchmark examples.
    
    Args:
        examples: List of benchmark examples to evaluate
        api_key: OpenAI API key
        model_id: Model ID to use
        results_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating ToP method on {len(examples)} examples")
    
    # Initialize Game24Solver with ToP framework
    solver = Game24Solver(
        api_key=api_key,
        model_id=model_id,
        results_dir=results_dir
    )
    
    # Track results
    results = []
    successful = 0
    total_time = 0
    
    # Evaluate each example
    for i, numbers in enumerate(examples):
        start_time = time.time()
        try:
            logger.info(f"Evaluating ToP method on example {i+1}/{len(examples)}: {numbers}")
            
            # Solve using ToP framework asynchronously
            async with logger.create_timed_section(f"top_example_{i}"):
                # Ensure we're properly awaiting the async solve method
                result = await solver.solve(numbers)
                
                # Add debug logging to see the raw result
                logger.debug(f"Raw solver result for {numbers}: {result}")
            
            # Verify solution correctness (in addition to solver's verification)
            solution = result.get("solution", "")
            is_correct = verify_solution(numbers, solution)
            
            # Update result with external verification
            result["externally_verified"] = is_correct
            
            # Calculate time
            solving_time = time.time() - start_time
            result["solving_time"] = solving_time
            
            # Record result
            results.append(result)
            
            if is_correct:
                successful += 1
                
            total_time += solving_time
            
            logger.info(f"ToP example {i+1}: " +
                     f"correct={is_correct}, time={solving_time:.2f}s, " +
                     f"solution={solution[:50]}...")
            
        except Exception as e:
            logger.error(f"Error evaluating ToP on example {i+1}: {e}", exc_info=True)
            logger.debug(f"Exception details for example {i+1}:", exc_info=True)
            solving_time = time.time() - start_time
            results.append({
                "numbers": numbers,
                "method": "top",
                "error": str(e),
                "is_correct": False,
                "externally_verified": False,
                "solving_time": solving_time
            })
    
    # Calculate metrics
    success_rate = successful / len(examples) if examples else 0
    avg_time = total_time / len(examples) if examples else 0
    
    logger.info(f"ToP evaluation completed: " +
             f"{successful}/{len(examples)} successful ({success_rate:.2%}), " +
             f"avg time: {avg_time:.2f}s")
    
    return {
        "method": "top",
        "total_examples": len(examples),
        "successful": successful,
        "success_rate": success_rate,
        "average_solving_time": avg_time,
        "results": results
    }


def save_evaluation_results(results: Dict[str, Any], results_dir: str) -> str:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Dictionary with evaluation results
        results_dir: Directory to save results
        
    Returns:
        Path to the saved results file
    """
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game24_evaluation_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Add timestamp to results
    results["timestamp"] = timestamp
    
    # Save results
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved evaluation results to {filepath}")
    return filepath


def compute_comparative_metrics(baseline_results: Dict[str, Any], top_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comparative metrics between baseline and ToP methods.
    
    Args:
        baseline_results: Results from baseline evaluation
        top_results: Results from ToP evaluation
        
    Returns:
        Dictionary with comparative metrics
    """
    # Extract key metrics
    baseline_success_rate = baseline_results.get("success_rate", 0)
    top_success_rate = top_results.get("success_rate", 0)
    
    baseline_avg_time = baseline_results.get("average_solving_time", 0)
    top_avg_time = top_results.get("average_solving_time", 0)
    
    # Calculate improvements
    success_rate_improvement = top_success_rate - baseline_success_rate
    success_rate_relative_improvement = (
        (top_success_rate / baseline_success_rate - 1) * 100 
        if baseline_success_rate > 0 else float('inf')
    )
    
    time_improvement = baseline_avg_time - top_avg_time
    time_relative_improvement = (
        (baseline_avg_time / top_avg_time - 1) * 100 
        if top_avg_time > 0 else float('inf')
    )
    
    # Find examples where methods differ
    baseline_results_dict = {
        tuple(result["numbers"]): result 
        for result in baseline_results.get("results", [])
    }
    
    top_results_dict = {
        tuple(result["numbers"]): result 
        for result in top_results.get("results", [])
    }
    
    # Find examples where ToP succeeded but baseline failed
    top_only_success = []
    for numbers, result in top_results_dict.items():
        if result.get("externally_verified", False):
            baseline_result = baseline_results_dict.get(numbers, {})
            if not baseline_result.get("is_correct", False):
                top_only_success.append({
                    "numbers": list(numbers),
                    "top_solution": result.get("solution", ""),
                    "baseline_solution": baseline_result.get("solution", "")
                })
    
    # Find examples where baseline succeeded but ToP failed
    baseline_only_success = []
    for numbers, result in baseline_results_dict.items():
        if result.get("is_correct", False):
            top_result = top_results_dict.get(numbers, {})
            if not top_result.get("externally_verified", False):
                baseline_only_success.append({
                    "numbers": list(numbers),
                    "baseline_solution": result.get("solution", ""),
                    "top_solution": top_result.get("solution", "")
                })
    
    return {
        "success_rate_comparison": {
            "baseline": baseline_success_rate,
            "top": top_success_rate,
            "absolute_improvement": success_rate_improvement,
            "relative_improvement_percent": success_rate_relative_improvement
        },
        "solving_time_comparison": {
            "baseline": baseline_avg_time,
            "top": top_avg_time,
            "absolute_improvement": time_improvement,
            "relative_improvement_percent": time_relative_improvement
        },
        "differential_analysis": {
            "top_only_success_count": len(top_only_success),
            "baseline_only_success_count": len(baseline_only_success),
            "top_only_success_examples": top_only_success[:10],  # Limit to 10 examples
            "baseline_only_success_examples": baseline_only_success[:10]  # Limit to 10 examples
        }
    }


async def main_async():
    """Async main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set up experiment logging
    logger.log_experiment_start("game24_evaluation", vars(args))
    logger.info("Starting evaluation with asyncio support")
    
    try:
        # Load benchmark data
        examples = load_benchmark_data(
            args.benchmark_path, 
            args.sample_size, 
            args.random_seed
        )
        
        # Evaluate methods based on flags
        baseline_results = None
        top_results = None
        
        if not args.top_only:
            # Evaluate baseline BFS method (this is not async)
            logger.info("Starting baseline BFS evaluation")
            baseline_results = evaluate_baseline_bfs(
                examples, 
                api_key=args.api_key,
                model_id=args.baseline_model_id
            )
        
        if not args.baseline_only:
            # Evaluate ToP method asynchronously
            logger.info("Starting ToP evaluation (async)")
            top_results = await evaluate_top_method(
                examples, 
                api_key=args.api_key,
                model_id=args.model_id,
                results_dir=args.results_dir
            )
        
        # Compute comparative metrics if both methods were evaluated
        comparative_metrics = None
        if baseline_results and top_results:
            comparative_metrics = compute_comparative_metrics(baseline_results, top_results)
        
        # Prepare final results
        final_results = {
            "configuration": vars(args),
            "benchmark_info": {
                "total_examples": len(examples),
                "path": args.benchmark_path,
                "sample_size": args.sample_size if args.sample_size > 0 else len(examples)
            }
        }
        
        if baseline_results:
            final_results["baseline_results"] = baseline_results
            
        if top_results:
            final_results["top_results"] = top_results
            
        if comparative_metrics:
            final_results["comparative_analysis"] = comparative_metrics
        
        # Save results
        results_path = save_evaluation_results(final_results, args.results_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nBenchmark: {args.benchmark_path}")
        print(f"Examples evaluated: {len(examples)}")
        
        if baseline_results:
            print(f"\nBaseline BFS method:")
            print(f"  Success rate: {baseline_results['success_rate']:.2%}")
            print(f"  Average solving time: {baseline_results['average_solving_time']:.2f}s")
        
        if top_results:
            print(f"\nToP method:")
            print(f"  Success rate: {top_results['success_rate']:.2%}")
            print(f"  Average solving time: {top_results['average_solving_time']:.2f}s")
        
        if comparative_metrics:
            sr_imp = comparative_metrics["success_rate_comparison"]["absolute_improvement"]
            sr_rel_imp = comparative_metrics["success_rate_comparison"]["relative_improvement_percent"]
            
            time_imp = comparative_metrics["solving_time_comparison"]["absolute_improvement"]
            time_rel_imp = comparative_metrics["solving_time_comparison"]["relative_improvement_percent"]
            
            print(f"\nImprovements:")
            print(f"  Success rate: {sr_imp:.2%} absolute, {sr_rel_imp:.2f}% relative")
            print(f"  Solving time: {time_imp:.2f}s absolute, {time_rel_imp:.2f}% relative")
            
            print(f"\nDifferential analysis:")
            print(f"  Examples where ToP succeeded but baseline failed: {comparative_metrics['differential_analysis']['top_only_success_count']}")
            print(f"  Examples where baseline succeeded but ToP failed: {comparative_metrics['differential_analysis']['baseline_only_success_count']}")
        
        print(f"\nDetailed results saved to: {results_path}")
        print("="*80 + "\n")
        
        # Log experiment end
        logger.log_experiment_end("game24_evaluation", final_results)
        
    except Exception as e:
        logger.critical(f"Evaluation failed: {e}", exc_info=True)
        raise


def main():
    """Main function to run the async evaluation."""
    try:
        # Set debug logging for asyncio to help diagnose issues
        if logger.isEnabledFor(logging.DEBUG):
            asyncio.get_event_loop().set_debug(True)
            
        # Run the async main function with proper asyncio handling
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
    except asyncio.CancelledError:
        logger.warning("Async operation was cancelled")
    except Exception as e:
        logger.critical(f"Evaluation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
