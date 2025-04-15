"""
Template Buffer for Template-oriented Prompting (ToP)

This module implements a specialized buffer for storing and retrieving solution templates,
with capabilities for dynamic updates based on template effectiveness and novelty.
It provides structured storage of templates with metadata and supports retrieving
templates based on semantic similarity to new problems.
"""

import os
import json
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

# For embedding and similarity calculation
from openai import OpenAI
from api_client import ApiClient
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("template_buffer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("template_buffer")


@dataclass
class TemplateMetadata:
    """Metadata for a solution template."""
    template_id: str  # Unique identifier
    problem_type: str  # Type of problem (e.g., 'game24', 'math', 'logic')
    created_at: str  # ISO format timestamp
    last_used: str  # ISO format timestamp
    use_count: int = 0  # Number of times used
    success_count: int = 0  # Number of successful uses
    tags: List[str] = field(default_factory=list)  # Tags for categorization
    description: str = ""  # Brief description of the template approach


@dataclass
class Template:
    """A solution template with content and metadata."""
    content: str  # The actual template content
    metadata: TemplateMetadata  # Associated metadata
    embedding: Optional[List[float]] = None  # Vector embedding for similarity search
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "content": self.content,
            "metadata": asdict(self.metadata)
        }
        if self.embedding is not None:
            result["embedding"] = self.embedding
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Template':
        """Create a Template from a dictionary."""
        metadata = TemplateMetadata(**data["metadata"])
        embedding = data.get("embedding")
        return cls(content=data["content"], metadata=metadata, embedding=embedding)


class TemplateBuffer:
    """
    Buffer for storing and retrieving solution templates.
    
    The TemplateBuffer maintains a collection of solution templates with their associated
    metadata and embeddings. It supports retrieving templates based on semantic similarity,
    updating templates based on their effectiveness, and dynamically adding new templates
    when they offer novel solution approaches.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4-turbo",
        base_url: str = "https://api.openai.com/v1/",
        buffer_dir: str = "./templates",
        embedding_dimension: int = 3072,
        similarity_threshold: float = 0.75,
        success_rate_threshold: float = 0.5
    ):
        """
        Initialize the TemplateBuffer.
        
        Args:
            api_key: OpenAI API key (optional, can use environment variable)
            embedding_model: Model to use for generating embeddings
            llm_model: Model to use for template assessment
            base_url: Base URL for API calls
            buffer_dir: Directory to store template data
            embedding_dimension: Dimension of embedding vectors
            similarity_threshold: Threshold for considering templates similar
            success_rate_threshold: Threshold for keeping templates during pruning
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Using environment variable.")
            
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.base_url = base_url
        self.buffer_dir = buffer_dir
        self.embedding_dimension = embedding_dimension
        self.similarity_threshold = similarity_threshold
        self.success_rate_threshold = success_rate_threshold
        
        # Create directories if they don't exist
        if not os.path.exists(buffer_dir):
            os.makedirs(buffer_dir)
            logger.info(f"Created template buffer directory: {buffer_dir}")
            
        # Initialize client for API calls
        self.api_client = ApiClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.llm_model
        )
        
        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Load existing templates
        self.templates: Dict[str, Template] = {}
        self.load_templates()
        
        logger.info(f"Initialized TemplateBuffer with {len(self.templates)} templates")
    
    def load_templates(self):
        """Load templates from disk."""
        template_file = os.path.join(self.buffer_dir, "templates.json")
        if os.path.exists(template_file):
            try:
                with open(template_file, "r") as f:
                    templates_data = json.load(f)
                
                for template_data in templates_data:
                    template = Template.from_dict(template_data)
                    self.templates[template.metadata.template_id] = template
                
                logger.info(f"Loaded {len(self.templates)} templates from {template_file}")
            except Exception as e:
                logger.error(f"Error loading templates: {e}")
        else:
            logger.info(f"No template file found at {template_file}, starting with empty buffer")
    
    def save_templates(self):
        """Save templates to disk."""
        template_file = os.path.join(self.buffer_dir, "templates.json")
        try:
            templates_data = [template.to_dict() for template in self.templates.values()]
            with open(template_file, "w") as f:
                json.dump(templates_data, f, indent=2)
            
            logger.info(f"Saved {len(self.templates)} templates to {template_file}")
        except Exception as e:
            logger.error(f"Error saving templates: {e}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a text using the OpenAI API.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not embedding1 or not embedding2:
            return 0.0
            
        # Convert to numpy arrays for efficient computation
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        
        # Compute cosine similarity: dot product / (norm(e1) * norm(e2))
        dot_product = np.dot(e1, e2)
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        
        if norm_e1 == 0 or norm_e2 == 0:
            return 0.0
            
        return dot_product / (norm_e1 * norm_e2)
    
    async def add_template(
        self, 
        content: str, 
        problem_type: str, 
        description: str = "", 
        tags: List[str] = None
    ) -> str:
        """
        Add a new template to the buffer.
        
        Args:
            content: The template content
            problem_type: Type of problem this template solves
            description: Brief description of the template
            tags: List of tags for categorization
            
        Returns:
            The ID of the newly added template
        """
        template_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        # Create metadata
        metadata = TemplateMetadata(
            template_id=template_id,
            problem_type=problem_type,
            created_at=current_time,
            last_used=current_time,
            tags=tags or [],
            description=description
        )
        
        # Get embedding
        embedding = await self.get_embedding(content)
        
        # Create and store template
        template = Template(
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        self.templates[template_id] = template
        logger.info(f"Added new template {template_id} of type {problem_type}")
        
        # Save to disk
        self.save_templates()
        
        return template_id
    
    async def retrieve_templates(
        self, 
        problem_description: str, 
        problem_type: Optional[str] = None, 
        top_k: int = 3
    ) -> List[Template]:
        """
        Retrieve the most relevant templates for a given problem.
        
        Args:
            problem_description: Description of the problem to solve
            problem_type: Optional filter by problem type
            top_k: Number of top templates to return
            
        Returns:
            List of the most relevant templates
        """
        if not self.templates:
            logger.warning("No templates available in buffer")
            return []
        
        start_time = time.time()
        
        # Get embedding for the problem description
        query_embedding = await self.get_embedding(problem_description)
        
        # Filter by problem type if specified
        candidates = self.templates.values()
        if problem_type:
            candidates = [t for t in candidates if t.metadata.problem_type == problem_type]
        
        # Compute similarities and sort
        similarities = []
        for template in candidates:
            if template.embedding:
                similarity = self.compute_similarity(query_embedding, template.embedding)
                similarities.append((template, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k templates
        top_templates = [t for t, s in similarities[:top_k]]
        
        # Update usage statistics for retrieved templates
        current_time = datetime.now().isoformat()
        for template in top_templates:
            template.metadata.last_used = current_time
            template.metadata.use_count += 1
        
        end_time = time.time()
        logger.info(f"Retrieved {len(top_templates)} templates in {end_time - start_time:.2f}s")
        
        # Log the top template and its similarity
        if similarities:
            top_template, top_similarity = similarities[0]
            logger.info(f"Top template: {top_template.metadata.template_id}, similarity: {top_similarity:.4f}")
        
        return top_templates
    
    def update_template_success(self, template_id: str, success: bool = True) -> bool:
        """
        Update template success statistics.
        
        Args:
            template_id: ID of the template to update
            success: Whether the template was successfully applied
            
        Returns:
            Boolean indicating success of the update operation
        """
        if template_id not in self.templates:
            logger.warning(f"Template {template_id} not found for updating")
            return False
        
        template = self.templates[template_id]
        if success:
            template.metadata.success_count += 1
        
        logger.info(f"Updated template {template_id} success status: {success}")
        self.save_templates()
        return True
    
    async def is_novel_template(self, new_template_content: str, problem_type: str) -> bool:
        """
        Determine if a new template is significantly different from existing ones.
        
        Args:
            new_template_content: Content of the proposed new template
            problem_type: Type of problem the template solves
            
        Returns:
            Boolean indicating whether the template is novel
        """
        # Only compare with templates of the same problem type
        existing_templates = [t for t in self.templates.values() 
                             if t.metadata.problem_type == problem_type]
        
        if not existing_templates:
            logger.info(f"No existing templates of type {problem_type}, considering new template as novel")
            return True
        
        # Get embedding for the new template
        new_embedding = await self.get_embedding(new_template_content)
        
        # Compute similarities with existing templates
        max_similarity = 0.0
        most_similar_template_id = None
        
        for template in existing_templates:
            if template.embedding:
                similarity = self.compute_similarity(new_embedding, template.embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_template_id = template.metadata.template_id
        
        is_novel = max_similarity < self.similarity_threshold
        
        if is_novel:
            logger.info(f"New template is novel (max similarity: {max_similarity:.4f})")
        else:
            logger.info(f"New template is similar to existing template {most_similar_template_id} "
                       f"(similarity: {max_similarity:.4f})")
        
        return is_novel
    
    async def assess_template_novelty(
        self, 
        new_template_content: str, 
        problem_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Assess if a template is novel using both embedding similarity and LLM assessment.
        
        Args:
            new_template_content: Content of the proposed new template
            problem_type: Type of problem the template solves
            
        Returns:
            Tuple of (is_novel, explanation)
        """
        # First check using embeddings
        embedding_novel = await self.is_novel_template(new_template_content, problem_type)
        
        if not embedding_novel:
            return False, "Template is too similar to existing templates based on embedding comparison"
        
        # Find most similar template for LLM comparison
        existing_templates = [t for t in self.templates.values() 
                             if t.metadata.problem_type == problem_type]
        
        if not existing_templates:
            return True, "No existing templates to compare with"
        
        # Get embedding for the new template
        new_embedding = await self.get_embedding(new_template_content)
        
        # Find most similar template
        most_similar_template = None
        max_similarity = -1.0
        
        for template in existing_templates:
            if template.embedding:
                similarity = self.compute_similarity(new_embedding, template.embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_template = template
        
        # Now use LLM to assess novelty
        if most_similar_template:
            prompt = f"""
You are an expert in analyzing problem-solving strategies. Determine if the new template 
represents a novel approach compared to the most similar existing template.

New Template:
{new_template_content}

Most Similar Existing Template:
{most_similar_template.content}

Please analyze both templates and determine:
1. If there is a fundamental difference in the problem-solving approach
2. If the new template offers any advantages in terms of efficiency or applicability
3. If the new template can solve cases that the existing template cannot

Based on your analysis, provide a binary decision:
- Output "True" if the new template represents a novel approach worth adding to the collection
- Output "False" if the new template is redundant or too similar to existing templates

Begin your response with the binary decision (True/False) followed by your reasoning.
"""
            
            response = self.api_client.generate(prompt)
            
            # Extract the binary decision
            is_novel = response.strip().lower().startswith("true")
            
            logger.info(f"LLM assessment of template novelty: {is_novel}")
            return is_novel, response
        else:
            return True, "No similar template found for LLM comparison"
    
    def prune_templates(self) -> int:
        """
        Remove redundant or ineffective templates.
        
        Returns:
            Number of templates removed
        """
        # Identify candidates for removal
        candidates_for_removal = []
        
        for template_id, template in self.templates.items():
            # Skip templates with no usage
            if template.metadata.use_count == 0:
                continue
                
            # Calculate success rate
            success_rate = template.metadata.success_count / template.metadata.use_count
            
            # Mark for removal if below threshold and used enough times
            if (success_rate < self.success_rate_threshold and 
                template.metadata.use_count >= 5):
                candidates_for_removal.append(template_id)
        
        # Remove the identified templates
        for template_id in candidates_for_removal:
            del self.templates[template_id]
        
        if candidates_for_removal:
            logger.info(f"Pruned {len(candidates_for_removal)} ineffective templates")
            self.save_templates()
        
        return len(candidates_for_removal)
    
    async def dynamic_update(
        self, 
        new_template_content: str, 
        problem_type: str, 
        description: str = "", 
        tags: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Dynamically update the template buffer with a new template if it's novel.
        
        Args:
            new_template_content: Content of the proposed new template
            problem_type: Type of problem the template solves
            description: Brief description of the template
            tags: List of tags for categorization
            
        Returns:
            Tuple of (was_added, template_id_or_reason)
        """
        # Assess if the template is novel
        is_novel, explanation = await self.assess_template_novelty(new_template_content, problem_type)
        
        if is_novel:
            # Add the template if it's novel
            template_id = await self.add_template(
                content=new_template_content,
                problem_type=problem_type,
                description=description,
                tags=tags
            )
            logger.info(f"Added novel template {template_id}")
            return True, template_id
        else:
            logger.info(f"Rejected non-novel template: {explanation}")
            return False, explanation
    
    def get_template_by_id(self, template_id: str) -> Optional[Template]:
        """
        Get a template by its ID.
        
        Args:
            template_id: ID of the template to retrieve
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(template_id)
    
    def get_templates_by_problem_type(self, problem_type: str) -> List[Template]:
        """
        Get all templates for a specific problem type.
        
        Args:
            problem_type: Type of problem
            
        Returns:
            List of templates for the given problem type
        """
        return [t for t in self.templates.values() if t.metadata.problem_type == problem_type]
    
    def get_template_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the templates in the buffer.
        
        Returns:
            Dictionary with statistics
        """
        problem_types = {}
        total_templates = len(self.templates)
        total_uses = 0
        total_successes = 0
        
        for template in self.templates.values():
            problem_type = template.metadata.problem_type
            problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
            total_uses += template.metadata.use_count
            total_successes += template.metadata.success_count
        
        success_rate = total_successes / total_uses if total_uses > 0 else 0
        
        return {
            "total_templates": total_templates,
            "problem_types": problem_types,
            "total_uses": total_uses,
            "total_successes": total_successes,
            "overall_success_rate": success_rate
        }

    async def retrieve_and_instantiate(
        self, 
        problem_description: str, 
        problem_type: Optional[str] = None,
        top_k: int = 3
    ) -> Tuple[Optional[Template], str]:
        """
        Retrieve relevant templates and instantiate the best one for the given problem.
        
        Args:
            problem_description: Description of the problem to solve
            problem_type: Optional filter by problem type
            top_k: Number of top templates to consider
            
        Returns:
            Tuple of (selected_template, instantiated_solution)
        """
        # Retrieve relevant templates
        templates = await self.retrieve_templates(
            problem_description=problem_description,
            problem_type=problem_type,
            top_k=top_k
        )
        
        if not templates:
            logger.warning("No templates found for instantiation")
            return None, "No suitable templates found"
        
        # Use the top template
        selected_template = templates[0]
        
        # Create prompt for template instantiation
        instantiation_prompt = f"""
You are an expert in problem-solving. Given a problem description and a solution template,
your task is to instantiate the template to solve the specific problem.

Problem description:
{problem_description}

Solution template:
{selected_template.content}

Please instantiate this template to solve the given problem:
1. Identify the key elements from the problem that map to the template
2. Adapt the template's approach to address the specific details of this problem
3. Provide a complete and detailed solution

Your response should be a fully instantiated solution that follows the template's approach
but is specifically tailored to solve the given problem.
"""
        
        # Generate the instantiated solution
        instantiated_solution = self.api_client.generate(instantiation_prompt)
        
        # Update template usage
        selected_template.metadata.use_count += 1
        selected_template.metadata.last_used = datetime.now().isoformat()
        self.save_templates()
        
        logger.info(f"Instantiated template {selected_template.metadata.template_id} for problem")
        
        return selected_template, instantiated_solution