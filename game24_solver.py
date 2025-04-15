"""
Game of 24 solver using Template-oriented Prompting (ToP) framework.

This module implements a solver for the Game of 24 using the ToP framework, which combines
template-based reasoning with dynamic error correction. The solver builds on a baseline
BFS method by retrieving relevant templates from a buffer, instantiating them for specific
problems, detecting and correcting errors, and updating the buffer with successful solutions.
"""

import re
import json
import os
import time
import logging
import asyncio
from typing import List, Dict, Optional, Union, Any, Coroutine
from datetime import datetime

# Import utilities
from meta_buffer_utilis import extract_and_execute_code
from api_client import ApiClient
from prompt_templates import game24_templates
from template_buffer import TemplateBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("game24_solver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("game24_solver")


class Game24Solver:
    """
    Game of 24 solver that leverages the ToP framework to solve arithmetic problems.
    
    This solver combines template-oriented prompting with a baseline BFS method to 
    efficiently find solutions to Game of 24 problems. It follows the ToP framework's
    key components: problem distillation, template retrieval, solution instantiation,
    error detection and correction, and dynamic buffer updates.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_id: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-large",
        base_url: str = "https://api.openai.com/v1/",
        buffer_dir: str = "./templates",
        results_dir: str = "./results"
    ):
        """
        Initialize the Game of 24 solver.
        
        Args:
            api_key: OpenAI API key (optional, can use environment variable)
            model_id: Model to use for generation
            embedding_model: Model to use for embeddings
            base_url: Base URL for API calls
            buffer_dir: Directory for template storage
            results_dir: Directory for saving results
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_id = model_id
        self.embedding_model = embedding_model
        self.base_url = base_url
        
        # Initialize API client for LLM interactions
        self.api_client = ApiClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_id
        )
        
        # Initialize template buffer
        self.template_buffer = TemplateBuffer(
            api_key=self.api_key,
            embedding_model=self.embedding_model,
            llm_model=self.model_id,
            base_url=self.base_url,
            buffer_dir=buffer_dir
        )
        
        # Get Game of 24 prompt templates
        self.templates = game24_templates
        
        # Create results directory if it doesn't exist
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Initialize baseline BFS template - defer to an async method that will be called later
        self._baseline_initialized = False
        
        logger.info(f"Initialized Game24Solver with model {model_id}")
        
    async def initialize(self):
        """
        Async initialization method to be called after constructor.
        This handles any async setup operations.
        """
        if not self._baseline_initialized:
            await self._initialize_baseline()
            self._baseline_initialized = True
        
    async def _initialize_baseline(self):
        """Initialize and store the baseline BFS approach in the template buffer."""
        # Baseline BFS template for Game of 24
        baseline_template = """
from itertools import permutations, product

def find_solution(numbers):
    # Find a solution to the Game of 24 using BFS over all possible expressions
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
                    result = eval(expr)
                    if abs(result - 24) < 1e-6:
                        return expr
                except (ZeroDivisionError, SyntaxError):
                    continue
    return "No solution found."

# Example usage
numbers = [NUMBERS]
solution = find_solution(numbers)
print(solution)
"""
        
        try:
            # Check if baseline already exists in the buffer
            templates = await self.template_buffer.get_templates_by_problem_type("game24")
            baseline_exists = any("baseline" in (t.metadata.tags or []) for t in templates)
            
            if not baseline_exists:
                logger.info("Adding baseline BFS template to template buffer")
                await self.template_buffer.add_template(
                    content=baseline_template,
                    problem_type="game24",
                    description="Baseline BFS approach for Game of 24",
                    tags=["baseline", "bfs", "game24"]
                )
                logger.info("Successfully added baseline template to buffer")
        except Exception as e:
            logger.error(f"Error initializing baseline template: {str(e)}")
            # Create a fallback template in memory if we can't access the buffer
            logger.info("Using in-memory fallback for baseline template")

    async def distill_problem(self, numbers: List[int]) -> str:
        """
        Distill the Game of 24 problem to extract key information.
        
        Args:
            numbers: List of four integers for the Game of 24 problem
            
        Returns:
            Distilled problem description
        """
        prompt = self.templates.get_distillation_prompt()
        problem = f"Find a way to make 24 using the numbers {numbers} with basic arithmetic operations."
        
        logger.info(f"Distilling problem for numbers: {numbers}")
        
        try:
            response = await self.api_client.generate(
                prompt=prompt + "\n\n" + problem,
                system_prompt="You are a mathematical problem analyzer specialized in Game of 24 problems.",
                temperature=0.2
            )
            
            distilled_problem = f"Game of 24 with numbers {numbers}: {response}"
            logger.debug(f"Distilled problem: {distilled_problem[:100]}...")
            
            return distilled_problem
        except Exception as e:
            logger.error(f"Error in distill_problem: {str(e)}")
            return f"Game of 24 with numbers {numbers}: [Error: {str(e)}]"

    async def retrieve_template(self, distilled_problem: str) -> Dict:
        """
        Retrieve the most relevant template for solving the problem.
        
        Args:
            distilled_problem: Distilled description of the problem
            
        Returns:
            Dictionary containing the selected template and its metadata
        """
        logger.info("Retrieving template from buffer")
        
        try:
            templates = await self.template_buffer.retrieve_templates(
                problem_description=distilled_problem,
                problem_type="game24",
                top_k=3
            )
        except Exception as e:
            logger.error(f"Error retrieving templates: {str(e)}")
            templates = []
        
        if not templates:
            logger.warning("No templates found, using fallback baseline approach")
            # Use fallback baseline template
            baseline_template = """
from itertools import permutations, product

def find_solution(numbers):
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
                    result = eval(expr)
                    if abs(result - 24) < 1e-6:
                        return expr
                except (ZeroDivisionError, SyntaxError):
                    continue
    return "No solution found."

# Example usage
numbers = [NUMBERS]
solution = find_solution(numbers)
print(solution)
"""
            return {
                "template": baseline_template,
                "template_id": "baseline_fallback",
                "is_fallback": True
            }
        
        selected_template = templates[0]
        logger.info(f"Selected template: {selected_template.metadata.template_id}")
        
        return {
            "template": selected_template.content,
            "template_id": selected_template.metadata.template_id,
            "is_fallback": False
        }

    async def instantiate_solution(self, numbers: List[int], template_info: Dict) -> Dict:
        """
        Instantiate a solution based on the selected template.
        
        Args:
            numbers: List of four integers for the Game of 24 problem
            template_info: Dictionary containing template information
            
        Returns:
            Dictionary with instantiated solution and metadata
        """
        template = template_info["template"]
        # Replace placeholder with actual numbers
        template = template.replace("[NUMBERS]", str(numbers))
        
        # Log the template being used
        logger.debug(f"Using template with ID: {template_info['template_id']}")
        
        problem = f"Find a way to make 24 using the numbers {numbers} with basic arithmetic operations."
        
        logger.info(f"Instantiating solution for numbers: {numbers}")
        
        # Create prompt for instantiation
        instantiation_prompt = f"""
You are a Python expert tasked with adapting a template to solve a Game of 24 problem.

Problem: {problem}

Below is a template for solving Game of 24 problems. Please adapt it to solve the specific
problem with numbers {numbers}. Make sure the code is correct, handles all edge cases,
and follows best practices.

Template code:
```python
{template}
```

Provide ONLY the adapted Python code without any additional explanations or text.
The code should be complete and ready to execute.
"""
        
        try:
            response = await self.api_client.generate(instantiation_prompt, temperature=0.2)
            logger.debug(f"Instantiation response length: {len(response)}")
            
            # Extract and execute the code
            result, code = await extract_and_execute_code(response)
        except Exception as e:
            logger.error(f"Error in instantiate_solution: {str(e)}")
            return {
                "solution": "",
                "result": f"An error occurred: {str(e)}",
                "success": False
            }
        
        # Log the full code for debugging purposes
        logger.debug(f"Generated code:\n{code}")
        
        success = result and "No solution found" not in result and "An error occurred" not in result
        logger.info(f"Solution instantiation {'succeeded' if success else 'failed'}: {result}")
        
        return {
            "solution": code,
            "result": result,
            "success": success
        }

    async def detect_and_fix_errors(self, numbers: List[int], solution_info: Dict) -> Dict:
        """
        Detect and fix errors in the solution if needed.
        
        Args:
            numbers: List of four integers for the Game of 24 problem
            solution_info: Dictionary containing solution information
            
        Returns:
            Dictionary with fixed solution and metadata
        """
        if solution_info["success"]:
            logger.info("No error detection needed - solution is already successful")
            return solution_info
            
        logger.info(f"Detecting and fixing errors in solution for numbers: {numbers}")
        
        # Create prompt for error detection and correction
        fix_prompt = f"""
You are an expert Python programmer tasked with debugging a Game of 24 solver.

The problem: Find an expression that equals 24 using the numbers {numbers} exactly once,
with only addition, subtraction, multiplication, division, and parentheses.

The current code has an issue:
```python
{solution_info["solution"]}
```

Error or issue: {solution_info["result"]}

Please diagnose and fix the code. Ensure:
1. It correctly tries all valid combinations of operations and parentheses
2. It handles division by zero and other exceptions properly
3. It returns a valid expression that equals 24, or "No solution found" if impossible
4. The code is complete and self-contained

Provide ONLY the corrected Python code without any additional explanations.
"""
        
        try:
            response = await self.api_client.generate(fix_prompt, temperature=0.3)
            logger.debug(f"Error correction response length: {len(response)}")
            
            # Extract and execute the fixed code
            fixed_result, fixed_code = await extract_and_execute_code(response)
        except Exception as e:
            logger.error(f"Error in detect_and_fix_errors: {str(e)}")
            return {
                "solution": solution_info["solution"],
                "result": f"Error fixing solution: {str(e)}",
                "success": False,
                "fixed": False
            }
        
        # Log the full fixed code for debugging
        logger.debug(f"Fixed code:\n{fixed_code}")
        
        fixed_success = fixed_result and "No solution found" not in fixed_result and "An error occurred" not in fixed_result
        logger.info(f"Error correction {'succeeded' if fixed_success else 'failed'}: {fixed_result}")
        
        return {
            "solution": fixed_code,
            "result": fixed_result,
            "success": fixed_success,
            "fixed": True
        }

    def verify_solution(self, numbers: List[int], solution: str) -> bool:
        """
        Verify that a solution is correct for the Game of 24.
        
        Args:
            numbers: List of four integers for the Game of 24 problem
            solution: Solution expression to verify
            
        Returns:
            Boolean indicating whether the solution is valid
        """
        logger.info(f"Verifying solution: {solution}")
        
        # Extract expression if it contains an equals sign or explanation text
        if "=" in solution:
            expression = solution.split("=")[0].strip()
        elif "solution:" in solution.lower():
            expression = solution.split("solution:", 1)[1].strip()
        elif ":" in solution:
            # Handle other potential label formats
            expression = solution.split(":", 1)[1].strip()
        else:
            expression = solution.strip()
            
        # Remove any surrounding quotes or whitespace
        expression = expression.strip('"\'').strip()
        
        logger.debug(f"Extracted expression for verification: '{expression}'")
        
        # Extract all numbers from the expression
        # Handle multi-digit numbers correctly by using regex with word boundaries
        all_nums_in_expr = [int(n) for n in re.findall(r'\b\d+\b', expression)]
        
        # Check if all numbers from the input are used exactly once
        number_counts = {}
        for num in numbers:
            number_counts[num] = number_counts.get(num, 0) + 1
            
        expr_number_counts = {}
        for num in all_nums_in_expr:
            expr_number_counts[num] = expr_number_counts.get(num, 0) + 1
        
        # Log the extracted numbers for debugging
        logger.debug(f"Input numbers: {numbers}, counts: {number_counts}")
        logger.debug(f"Expression numbers: {all_nums_in_expr}, counts: {expr_number_counts}")
        
        # Check if all input numbers are used exactly once
        numbers_correct = True
        for num in numbers:
            if expr_number_counts.get(num, 0) != number_counts.get(num, 0):
                logger.warning(f"Number mismatch: {num} appears {number_counts.get(num)} times in input but {expr_number_counts.get(num, 0)} times in expression")
                numbers_correct = False
                break
                
        # Also check if any extra numbers appear in the expression
        for num in expr_number_counts:
            if num not in number_counts:
                logger.warning(f"Extra number in expression: {num} not in input numbers")
                numbers_correct = False
                break
                
        # Check if the expression evaluates to 24
        try:
            # Evaluate the expression directly - we want true division, not floor division
            result = eval(expression)
            value_correct = abs(result - 24) < 1e-6
            logger.debug(f"Expression '{expression}' evaluates to {result}")
        except Exception as e:
            logger.warning(f"Error evaluating expression '{expression}': {e}")
            value_correct = False
            
        is_valid = numbers_correct and value_correct
        logger.info(f"Solution verification: {is_valid} (numbers_correct: {numbers_correct}, value_correct: {value_correct})")
        
        return is_valid

    async def update_template_buffer(self, numbers: List[int], solution_info: Dict, template_info: Dict) -> None:
        """
        Update the template buffer with successful solutions.
        
        Args:
            numbers: List of four integers for the Game of 24 problem
            solution_info: Dictionary containing solution information
            template_info: Dictionary containing template information
        """
        if not solution_info["success"]:
            logger.info("Not updating template buffer - solution was not successful")
            return
            
        try:
            # Update success count for the used template if it's not a fallback
            if not template_info.get("is_fallback", False):
                await self.template_buffer.update_template_success(
                    template_id=template_info["template_id"],
                    success=True
                )
                logger.info(f"Updated success count for template {template_info['template_id']}")
            
            # If the solution was fixed, consider adding it as a new template
            if solution_info.get("fixed", False):
                logger.info("Considering fixed solution as potential new template")
                
                # Create a generalized version of the solution
                generalized_solution = solution_info["solution"].replace(str(numbers), "[NUMBERS]")
                
                # Check if the new template is novel
                is_novel, explanation = await self.template_buffer.assess_template_novelty(
                    new_template_content=generalized_solution,
                    problem_type="game24"
                )
                
                if is_novel:
                    logger.info("Adding novel template to buffer")
                    await self.template_buffer.add_template(
                        content=generalized_solution,
                        problem_type="game24",
                        description=f"Template derived from fixed solution for numbers {numbers}",
                        tags=["game24", "derived", "fixed"]
                    )
                else:
                    logger.info(f"Not adding template: {explanation}")
        except Exception as e:
            logger.error(f"Error updating template buffer: {str(e)}")

    async def solve(self, numbers: List[int]) -> Dict:
        """
        Solve a Game of 24 problem using the ToP framework.
        
        Args:
            numbers: List of four integers for the Game of 24 problem
            
        Returns:
            Dictionary with solution details and results
        """
        if len(numbers) != 4:
            return {"error": "Game of 24 requires exactly 4 numbers", "success": False}
        
        start_time = time.time()
        logger.info(f"Starting to solve Game of 24 with numbers: {numbers}")
        
        # Initialize variables to avoid reference errors in exception handling
        distilled_problem = None
        template_info = None
        solution_info = None
        
        try:
            # Step 1: Problem Distillation
            logger.debug("Starting problem distillation")
            distilled_problem = await self.distill_problem(numbers)
            logger.debug(f"Completed problem distillation: {distilled_problem[:50]}...")
            
            # Step 2: Template Retrieval
            logger.debug("Starting template retrieval")
            template_info = await self.retrieve_template(distilled_problem)
            logger.debug(f"Completed template retrieval: template_id={template_info['template_id']}")
            
            # Step 3: Solution Instantiation
            logger.debug("Starting solution instantiation")
            solution_info = await self.instantiate_solution(numbers, template_info)
            logger.debug(f"Completed solution instantiation: success={solution_info['success']}")
            
            # Step 4: Error Detection and Correction (if needed)
            if not solution_info["success"]:
                logger.debug("Starting error detection and correction")
                solution_info = await self.detect_and_fix_errors(numbers, solution_info)
                logger.debug(f"Completed error detection and correction: success={solution_info['success']}")
            
            # Step 5: Verification
            logger.debug("Starting solution verification")
            if solution_info["success"]:
                solution_info["verified"] = self.verify_solution(numbers, solution_info["result"])
            else:
                solution_info["verified"] = False
            logger.debug(f"Completed solution verification: verified={solution_info.get('verified', False)}")
                
            # Step 6: Dynamic Update of Template Buffer
            logger.debug("Starting template buffer update")
            await self.update_template_buffer(numbers, solution_info, template_info)
            logger.debug("Completed template buffer update")
        except Exception as e:
            logger.error(f"Error in solve method: {str(e)}")
            return {
                "numbers": numbers,
                "solution": f"Error: {str(e)}",
                "success": False,
                "solving_time": time.time() - start_time,
                "method": "Error",
                "template_id": template_info.get("template_id", "unknown") if 'template_info' in locals() else "unknown",
                "fixed": False,
                "verified": False,
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate solution time
        solving_time = time.time() - start_time
        
        # Prepare result
        result = {
            "numbers": numbers,
            "solution": solution_info["result"] if solution_info["success"] else "No solution found",
            "success": solution_info["success"] and solution_info["verified"],
            "solving_time": solving_time,
            "method": "ToP-enhanced" if not template_info.get("is_fallback", False) else "Fallback BFS",
            "template_id": template_info["template_id"],
            "fixed": solution_info.get("fixed", False),
            "verified": solution_info["verified"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save result to file
        self._save_result(result)
        
        logger.info(f"Completed solving Game of 24: success={result['success']}, time={solving_time:.2f}s")
        return result
        
    def _save_result(self, result: Dict) -> None:
        """
        Save the solution result to a JSON file.
        
        Args:
            result: Dictionary with solution details
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        numbers_str = "_".join(map(str, result["numbers"]))
        filename = f"game24_{numbers_str}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved result to {filepath}")

    async def batch_solve(self, batch: List[List[int]]) -> Dict:
        """
        Solve multiple Game of 24 problems and aggregate results.
        
        Args:
            batch: List of problems, where each problem is a list of four integers
            
        Returns:
            Dictionary with aggregated results and metrics
        """
        results = []
        successful = 0
        total_time = 0
        
        logger.info(f"Starting batch solve for {len(batch)} Game of 24 problems")
        
        for i, numbers in enumerate(batch):
            logger.info(f"Solving problem {i+1}/{len(batch)}: {numbers}")
            try:
                # Add more detailed logging
                logger.debug(f"Calling solve method for numbers: {numbers}")
                result = await self.solve(numbers)
                logger.debug(f"Solve method returned result with success={result.get('success', False)}")
                results.append(result)
                
                if result.get("success", False):
                    successful += 1
                total_time += result.get("solving_time", 0)
            except Exception as e:
                logger.error(f"Error solving problem {numbers}: {str(e)}")
                # Include traceback for better debugging
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                results.append({
                    "numbers": numbers,
                    "solution": f"Error: {str(e)}",
                    "success": False,
                    "solving_time": 0,
                    "method": "Error",
                    "template_id": "unknown",
                    "fixed": False,
                    "verified": False,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Aggregate metrics
        success_rate = successful / len(batch) if batch else 0
        avg_time = total_time / len(batch) if batch else 0
        
        # Prepare batch results
        batch_results = {
            "total_problems": len(batch),
            "successful_solutions": successful,
            "success_rate": success_rate,
            "average_solving_time": avg_time,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"game24_batch_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(batch_results, f, indent=2)
        
        logger.info(f"Batch solve completed: {successful}/{len(batch)} successful ({success_rate:.2%})")
        return batch_results


async def solve_game24(numbers: List[int], api_key: Optional[str] = None) -> Dict:
    """
    Convenience function to solve a Game of 24 problem.
    
    Args:
        numbers: List of four integers for the Game of 24 problem
        api_key: OpenAI API key (optional)
        
    Returns:
        Dictionary with solution details
    """
    try:
        # Create solver and initialize it properly
        solver = Game24Solver(api_key=api_key)
        await solver.initialize()
        
        # Solve the problem
        return await solver.solve(numbers)
    except Exception as e:
        logger.error(f"Error in solve_game24: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "numbers": numbers,
            "solution": f"Error: {str(e)}",
            "success": False,
            "solving_time": 0,
            "method": "Error",
            "template_id": "unknown",
            "fixed": False,
            "verified": False,
            "timestamp": datetime.now().isoformat()
        }

def solve_game24_sync(numbers: List[int], api_key: Optional[str] = None) -> Dict:
    """
    Synchronous wrapper for solve_game24.
    
    Args:
        numbers: List of four integers for the Game of 24 problem
        api_key: OpenAI API key (optional)
        
    Returns:
        Dictionary with solution details
    """
    try:
        return asyncio.run(solve_game24(numbers, api_key))
    except Exception as e:
        logger.error(f"Error in solve_game24_sync: {str(e)}")
        return {
            "numbers": numbers,
            "solution": f"Error: {str(e)}",
            "success": False,
            "solving_time": 0,
            "method": "Error",
            "template_id": "unknown",
            "fixed": False,
            "verified": False,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Solve Game of 24 using ToP framework')
    parser.add_argument('numbers', type=int, nargs=4, help='Four integers for the Game of 24')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        print("Debug logging enabled")
    
    # Use the synchronous wrapper for command-line usage
    print(f"Solving Game of 24 for numbers: {args.numbers}")
    result = solve_game24_sync(args.numbers, api_key=args.api_key)
    print(f"Numbers: {args.numbers}")
    print(f"Solution: {result['solution']}")
    print(f"Success: {result['success']}")
    print(f"Method: {result['method']}")
    print(f"Solving time: {result.get('solving_time', 0):.2f} seconds")
