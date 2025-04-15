"""
Prompt templates for the Template-oriented Prompting (ToP) framework.

This module contains a collection of prompt templates used in the ToP framework, 
specifically designed for the Game of 24 task. These templates are organized by their
role in the reasoning process: problem distillation, template retrieval, solution
instantiation, error detection, verification, and meta-buffer updates.
"""

from typing import List, Dict, Optional

# Problem Distillation Templates
# ------------------------------

GAME24_DISTILLATION_PROMPT = """
As an expert in mathematical problem analysis, your task is to distill the Game of 24 problem.

Given four integers, extract the essential information needed to solve the problem.
The goal is to use each number exactly once with the four basic arithmetic operations 
(addition, subtraction, multiplication, and division) and parentheses to obtain 24.

Please extract the following information:

1. Key information:
   - The four numbers that must be used exactly once.

2. Restriction:
   - Each number must be used exactly once.
   - Only +, -, *, / operations and parentheses are allowed.
   - The order of operations follows standard arithmetic rules.
   - The final result must equal exactly 24.

3. Distilled task:
   Find a way to combine the four numbers using the four operations to yield exactly 24.

4. Python transformation:
   Input parameters:
     numbers = [a, b, c, d]  # The four integers from the problem

5. Answer form:
   An expression using the four numbers that evaluates to 24, such as "(a + b) * c / d = 24"
"""

# Template Retrieval Templates
# ---------------------------

GAME24_RETRIEVAL_PROMPT = """
You are an expert in mathematical problem-solving with access to a library of solution templates.
Your task is to select the most appropriate template for solving a Game of 24 problem.

Game of 24 problem: {problem}

Evaluate each template in the library and select the one that best matches this problem:
1. Consider the structural similarity to the current problem
2. Evaluate the conceptual alignment between the template and this problem
3. Determine if the template's operations and approach are suitable

Return the template ID that would be most effective for solving this specific Game of 24 problem,
along with a brief explanation of why it's the best match.
"""

# Solution Instantiation Templates
# ------------------------------

GAME24_INSTANTIATION_PROMPT = """
As an expert in mathematical problem-solving, your task is to instantiate a solution for the Game of 24.

Problem: {problem}
Selected Template: {template}

Please follow these steps:
1. Map the four numbers from the problem to the template variables
2. Adapt the template's operations to work with these specific numbers
3. Verify that each number is used exactly once
4. Ensure that the operations follow standard arithmetic precedence rules
5. Check that the final result equals exactly 24

Your solution should be clear, showing step-by-step calculations and the final expression.
Include parentheses where necessary to ensure the correct order of operations.
"""

GAME24_PYTHON_INSTANTIATION_PROMPT = """
You are an expert in computational problem-solving. Given a Game of 24 problem and a solution template,
create a Python implementation that solves this specific instance.

Problem: {problem}
Template:
```python
{template}
```

Your task:
1. Carefully adapt the template code to work with the input numbers: {numbers}
2. Ensure the code is correct, efficient, and follows best practices
3. Verify that the solution is valid (uses each number exactly once and results in 24)
4. The code should return a valid expression that equals 24 using the input numbers

Provide ONLY the adapted Python code without any additional explanations.
The code should be complete and ready to execute.
"""

# Error Detection and Correction Templates
# --------------------------------------

GAME24_ERROR_DETECTION_PROMPT = """
As a mathematical verification expert, your task is to identify errors in the proposed solution
to the Game of 24 problem.

Problem: Use the numbers {numbers} to create an expression that equals 24.
Proposed Solution: {solution}

Please check for the following potential errors:
1. Arithmetic errors in calculations
2. Using a number more than once or not using a number
3. Using operations other than +, -, *, /
4. Incorrect application of order of operations
5. Expression not evaluating to exactly 24

For each error found, identify:
- The specific location of the error
- The nature of the error
- Why it's incorrect

If the solution is correct, confirm that it satisfies all Game of 24 requirements.
"""

GAME24_ERROR_CORRECTION_PROMPT = """
As a mathematical problem-solving expert, your task is to correct errors in the proposed solution
to the Game of 24 problem.

Problem: Use the numbers {numbers} to create an expression that equals 24.
Proposed Solution: {solution}
Detected Errors: {errors}

Please:
1. Propose a targeted fix for each identified error
2. Explain the reasoning behind each correction
3. Provide a revised solution that correctly solves the Game of 24 problem

The corrected solution must:
- Use each number exactly once
- Only use the operations +, -, *, /
- Follow standard order of operations
- Evaluate to exactly 24

Your response should include the corrected expression and confirmation that it evaluates to 24.
"""

# Verification Templates
# --------------------

GAME24_VERIFICATION_PROMPT = """
As a mathematical verification expert, your task is to verify the correctness of a solution
to the Game of 24 problem.

Problem: Use the numbers {numbers} to create an expression that equals 24.
Proposed Solution: {solution}

Please verify:
1. Each of the four numbers is used exactly once
2. Only the operations +, -, *, / and parentheses are used
3. The expression follows standard order of operations
4. When evaluated, the expression equals exactly 24

Step through the evaluation of the expression showing intermediate results.
Confirm whether the solution is valid or invalid with a clear explanation.
"""

# Meta-buffer Update Templates
# --------------------------

GAME24_TEMPLATE_EXTRACTION_PROMPT = """
As an expert in pattern recognition and knowledge distillation, analyze the following 
successful Game of 24 solution and extract a reusable solution template.

Problem: {problem}
Solution: {solution}

Create a generalized template that captures the solution strategy. Your template should:
1. Identify the key pattern or approach used in the solution
2. Abstract specific numbers into variables or placeholders
3. Preserve the structure of operations that led to success
4. Include any key insights that made finding the solution possible
5. Be applicable to a wide range of similar Game of 24 problems

Your output should include:
1. A formal template with variables instead of specific numbers
2. A description of when and how this template should be applied
3. Any mathematical properties or tricks that are leveraged
"""

GAME24_TEMPLATE_NOVELTY_PROMPT = """
As an expert in mathematical problem-solving strategies, your task is to determine if a newly 
discovered solution template for Game of 24 represents a novel approach compared to existing templates.

New Template:
{new_template}

Most Similar Existing Template:
{existing_template}

Please analyze both templates and determine:
1. If there is a fundamental difference in the problem-solving approach
2. If the new template offers any advantages in terms of efficiency or applicability
3. If the new template can solve cases that the existing template cannot

Based on your analysis, make a binary decision:
- Output "True" if the new template represents a novel approach worth adding to our collection
- Output "False" if the new template is redundant or too similar to existing templates

Justify your decision with a clear explanation of your reasoning.
"""

# Complete Template Collections
# ---------------------------

class Game24Templates:
    """
    Collection of prompt templates for the Game of 24 task within the ToP framework.
    
    This class organizes all prompt templates by their function in the reasoning process
    and provides methods to access and customize them.
    """
    
    def __init__(self):
        """Initialize the Game of 24 prompt templates."""
        self.distillation = GAME24_DISTILLATION_PROMPT
        self.retrieval = GAME24_RETRIEVAL_PROMPT
        self.instantiation = GAME24_INSTANTIATION_PROMPT
        self.python_instantiation = GAME24_PYTHON_INSTANTIATION_PROMPT
        self.error_detection = GAME24_ERROR_DETECTION_PROMPT
        self.error_correction = GAME24_ERROR_CORRECTION_PROMPT
        self.verification = GAME24_VERIFICATION_PROMPT
        self.template_extraction = GAME24_TEMPLATE_EXTRACTION_PROMPT
        self.template_novelty = GAME24_TEMPLATE_NOVELTY_PROMPT
    
    def get_distillation_prompt(self) -> str:
        """Get the problem distillation prompt."""
        return self.distillation
    
    def get_retrieval_prompt(self, problem: str) -> str:
        """
        Get the template retrieval prompt with the problem inserted.
        
        Args:
            problem (str): The Game of 24 problem description.
            
        Returns:
            str: The formatted template retrieval prompt.
        """
        return self.retrieval.format(problem=problem)
    
    def get_instantiation_prompt(self, problem: str, template: str) -> str:
        """
        Get the solution instantiation prompt with problem and template inserted.
        
        Args:
            problem (str): The Game of 24 problem description.
            template (str): The selected solution template.
            
        Returns:
            str: The formatted solution instantiation prompt.
        """
        return self.instantiation.format(problem=problem, template=template)
    
    def get_python_instantiation_prompt(self, problem: str, template: str, numbers: List[int]) -> str:
        """
        Get the Python solution instantiation prompt with all parameters inserted.
        
        Args:
            problem (str): The Game of 24 problem description.
            template (str): The selected Python solution template.
            numbers (List[int]): The four numbers for the Game of 24 problem.
            
        Returns:
            str: The formatted Python solution instantiation prompt.
        """
        return self.python_instantiation.format(
            problem=problem, 
            template=template,
            numbers=numbers
        )
    
    def get_error_detection_prompt(self, numbers: List[int], solution: str) -> str:
        """
        Get the error detection prompt with parameters inserted.
        
        Args:
            numbers (List[int]): The four numbers for the Game of 24 problem.
            solution (str): The proposed solution to verify.
            
        Returns:
            str: The formatted error detection prompt.
        """
        return self.error_detection.format(numbers=numbers, solution=solution)
    
    def get_error_correction_prompt(self, numbers: List[int], solution: str, errors: str) -> str:
        """
        Get the error correction prompt with parameters inserted.
        
        Args:
            numbers (List[int]): The four numbers for the Game of 24 problem.
            solution (str): The proposed solution to correct.
            errors (str): Description of detected errors.
            
        Returns:
            str: The formatted error correction prompt.
        """
        return self.error_correction.format(
            numbers=numbers, 
            solution=solution,
            errors=errors
        )
    
    def get_verification_prompt(self, numbers: List[int], solution: str) -> str:
        """
        Get the verification prompt with parameters inserted.
        
        Args:
            numbers (List[int]): The four numbers for the Game of 24 problem.
            solution (str): The proposed solution to verify.
            
        Returns:
            str: The formatted verification prompt.
        """
        return self.verification.format(numbers=numbers, solution=solution)
    
    def get_template_extraction_prompt(self, problem: str, solution: str) -> str:
        """
        Get the template extraction prompt with parameters inserted.
        
        Args:
            problem (str): The Game of 24 problem description.
            solution (str): The successful solution to extract a template from.
            
        Returns:
            str: The formatted template extraction prompt.
        """
        return self.template_extraction.format(problem=problem, solution=solution)
    
    def get_template_novelty_prompt(self, new_template: str, existing_template: str) -> str:
        """
        Get the template novelty assessment prompt with parameters inserted.
        
        Args:
            new_template (str): The newly discovered solution template.
            existing_template (str): The most similar existing template.
            
        Returns:
            str: The formatted template novelty assessment prompt.
        """
        return self.template_novelty.format(
            new_template=new_template, 
            existing_template=existing_template
        )

# Initialize the Game of 24 templates for easy import
game24_templates = Game24Templates()