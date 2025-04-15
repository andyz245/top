"""
Comprehensive logging utility for Template-oriented Prompting (ToP) framework.

This module provides logging capabilities for tracking inputs, outputs, and
intermediate reasoning steps during template-oriented prompting. It supports
structured logging for different components of the ToP framework, including
model interactions, template operations, and step-by-step reasoning processes.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class ToP_Logger:
    """
    Comprehensive logger for the Template-oriented Prompting (ToP) framework.
    
    This logger provides structured logging capabilities for tracking all aspects of the
    template-oriented prompting process, including model interactions, template operations,
    and reasoning steps. It supports both console and file logging with configurable
    verbosity levels.
    
    This class implements the standard logging.Logger interface for compatibility with
    libraries and frameworks that expect a standard logger.
    """
    
    def __init__(
        self,
        name: str = "top_logger",
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        enable_json_logs: bool = True,
        json_log_file: Optional[str] = None,
    ):
        """
        Initialize the ToP Logger.
        
        Args:
            name: Unique name for this logger instance
            log_dir: Directory where log files will be stored
            console_level: Logging level for console output (default: INFO)
            file_level: Logging level for file output (default: DEBUG)
            enable_json_logs: Whether to save structured JSON logs
            json_log_file: Name of the JSON log file (default: based on logger name)
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.enable_json_logs = enable_json_logs
        
        # Create log directory if it doesn't exist
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
        
        # Set up the standard Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
        
        # Clear any existing handlers (in case of re-initialization)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # Configure file handler for text logs
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Configure JSON logging
        self.json_log_file = json_log_file or f"{name}_structured.jsonl"
        self.json_log_path = self.log_dir / self.json_log_file
        
        # Initialize experiment context
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.context = {}
        
        # Log initialization
        self.logger.info(f"Initialized ToP Logger: {name} (ID: {self.experiment_id})")
    
    def set_context(self, **kwargs) -> None:
        """
        Set context variables that will be included in all subsequent log entries.
        
        Args:
            **kwargs: Key-value pairs to add to the context
        """
        self.context.update(kwargs)
        self.logger.debug(f"Updated context: {kwargs}")
    
    def clear_context(self) -> None:
        """Clear all context variables."""
        self.context = {}
        self.logger.debug("Cleared context")
    
    def _log_to_json(self, log_type: str, message: str, data: Dict[str, Any]) -> None:
        """
        Log structured data to JSON file.
        
        Args:
            log_type: Type of log entry (e.g., 'model_input', 'reasoning_step')
            message: Text message for this log entry
            data: Structured data to log
        """
        if not self.enable_json_logs:
            return
        
        # Skip JSON logging if the logger level would filter this message
        try:
            if log_type == "debug" and not self.isEnabledFor(logging.DEBUG):
                return
            if log_type == "info" and not self.isEnabledFor(logging.INFO):
                return
            if log_type == "warning" and not self.isEnabledFor(logging.WARNING):
                return
            if log_type == "error" and not self.isEnabledFor(logging.ERROR):
                return
            if log_type == "critical" and not self.isEnabledFor(logging.CRITICAL):
                return
        except AttributeError:
            # Fallback if isEnabledFor is not available
            pass
        
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "log_type": log_type,
            "message": message,
            **self.context,
            **data
        }
        
        # Write to JSON log file
        with open(self.json_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    # Standard logging.Logger interface methods
    def isEnabledFor(self, level: int) -> bool:
        """
        Check if this logger is enabled for the specified level.
        
        Args:
            level: The logging level to check
            
        Returns:
            True if the logger is enabled for the specified level, False otherwise
        """
        return self.logger.isEnabledFor(level)
    
    def getEffectiveLevel(self) -> int:
        """
        Get the effective level of this logger.
        
        Returns:
            The effective level of the logger
        """
        return self.logger.getEffectiveLevel()
    
    def setLevel(self, level: int) -> None:
        """
        Set the logging level of this logger.
        
        Args:
            level: The new logging level
        """
        self.logger.setLevel(level)
    
    # Additional standard Logger interface methods
    def addHandler(self, hdlr) -> None:
        """Add the specified handler to this logger."""
        self.logger.addHandler(hdlr)
    
    def removeHandler(self, hdlr) -> None:
        """Remove the specified handler from this logger."""
        self.logger.removeHandler(hdlr)
    
    def hasHandlers(self) -> bool:
        """See if this logger has any handlers configured."""
        return self.logger.hasHandlers()
    
    def callHandlers(self, record) -> None:
        """Pass a record to all relevant handlers."""
        self.logger.callHandlers(record)
    
    def handle(self, record) -> None:
        """Call the handlers for the specified record."""
        self.logger.handle(record)
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None) -> logging.LogRecord:
        """Make a LogRecord."""
        return self.logger.makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
    
    def findCaller(self, stack_info=False, stacklevel=1):
        """Find the caller's source file and line number."""
        return self.logger.findCaller(stack_info, stacklevel)
    
    @property
    def propagate(self) -> bool:
        """Get the propagate setting."""
        return self.logger.propagate
    
    @propagate.setter
    def propagate(self, value: bool) -> None:
        """Set the propagate setting."""
        self.logger.propagate = value
    
    @property
    def handlers(self):
        """Get the handlers for this logger."""
        return self.logger.handlers
    
    # General logging methods
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message with optional structured data."""
        extra_kwargs = {}
        if kwargs:
            # Extract standard logging kwargs
            exc_info = kwargs.pop('exc_info', None)
            stack_info = kwargs.pop('stack_info', None)
            stacklevel = kwargs.pop('stacklevel', None)
            extra = kwargs.pop('extra', {})
            
            # Pass standard kwargs to the logger
            extra_kwargs = {
                'exc_info': exc_info,
                'stack_info': stack_info,
                'stacklevel': stacklevel,
                'extra': extra
            }
            # Remove None values
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            
            # Log remaining kwargs to JSON
            if kwargs:
                self._log_to_json("debug", message, kwargs)
        
        self.logger.debug(message, *args, **extra_kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log an info message with optional structured data."""
        extra_kwargs = {}
        if kwargs:
            # Extract standard logging kwargs
            exc_info = kwargs.pop('exc_info', None)
            stack_info = kwargs.pop('stack_info', None)
            stacklevel = kwargs.pop('stacklevel', None)
            extra = kwargs.pop('extra', {})
            
            # Pass standard kwargs to the logger
            extra_kwargs = {
                'exc_info': exc_info,
                'stack_info': stack_info,
                'stacklevel': stacklevel,
                'extra': extra
            }
            # Remove None values
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            
            # Log remaining kwargs to JSON
            if kwargs:
                self._log_to_json("info", message, kwargs)
        
        self.logger.info(message, *args, **extra_kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message with optional structured data."""
        extra_kwargs = {}
        if kwargs:
            # Extract standard logging kwargs
            exc_info = kwargs.pop('exc_info', None)
            stack_info = kwargs.pop('stack_info', None)
            stacklevel = kwargs.pop('stacklevel', None)
            extra = kwargs.pop('extra', {})
            
            # Pass standard kwargs to the logger
            extra_kwargs = {
                'exc_info': exc_info,
                'stack_info': stack_info,
                'stacklevel': stacklevel,
                'extra': extra
            }
            # Remove None values
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            
            # Log remaining kwargs to JSON
            if kwargs:
                self._log_to_json("warning", message, kwargs)
        
        self.logger.warning(message, *args, **extra_kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log an error message with optional structured data."""
        extra_kwargs = {}
        if kwargs:
            # Extract standard logging kwargs
            exc_info = kwargs.pop('exc_info', None)
            stack_info = kwargs.pop('stack_info', None)
            stacklevel = kwargs.pop('stacklevel', None)
            extra = kwargs.pop('extra', {})
            
            # Pass standard kwargs to the logger
            extra_kwargs = {
                'exc_info': exc_info,
                'stack_info': stack_info,
                'stacklevel': stacklevel,
                'extra': extra
            }
            # Remove None values
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            
            # Log remaining kwargs to JSON
            if kwargs:
                self._log_to_json("error", message, kwargs)
        
        self.logger.error(message, *args, **extra_kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log a critical message with optional structured data."""
        extra_kwargs = {}
        if kwargs:
            # Extract standard logging kwargs
            exc_info = kwargs.pop('exc_info', None)
            stack_info = kwargs.pop('stack_info', None)
            stacklevel = kwargs.pop('stacklevel', None)
            extra = kwargs.pop('extra', {})
            
            # Pass standard kwargs to the logger
            extra_kwargs = {
                'exc_info': exc_info,
                'stack_info': stack_info,
                'stacklevel': stacklevel,
                'extra': extra
            }
            # Remove None values
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            
            # Log remaining kwargs to JSON
            if kwargs:
                self._log_to_json("critical", message, kwargs)
        
        self.logger.critical(message, *args, **extra_kwargs)
    
    # Aliases for standard logging methods
    def warn(self, message: str, *args, **kwargs) -> None:
        """Alias for warning() for compatibility with standard logger."""
        self.warning(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log an exception with traceback."""
        kwargs['exc_info'] = True
        self.error(message, *args, **kwargs)
    
    def log(self, level: int, message: str, *args, **kwargs) -> None:
        """Log a message with the specified level."""
        extra_kwargs = {}
        if kwargs:
            # Extract standard logging kwargs
            exc_info = kwargs.pop('exc_info', None)
            stack_info = kwargs.pop('stack_info', None)
            stacklevel = kwargs.pop('stacklevel', None)
            extra = kwargs.pop('extra', {})
            
            # Pass standard kwargs to the logger
            extra_kwargs = {
                'exc_info': exc_info,
                'stack_info': stack_info,
                'stacklevel': stacklevel,
                'extra': extra
            }
            # Remove None values
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            
            # Log remaining kwargs to JSON
            if kwargs:
                self._log_to_json(f"level_{level}", message, kwargs)
        
        self.logger.log(level, message, *args, **extra_kwargs)
    
    # Specialized logging methods for ToP framework
    def log_model_input(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log input to a language model.
        
        Args:
            prompt: The prompt text sent to the model
            system_prompt: Optional system prompt
            temperature: Model temperature setting
            max_tokens: Model max_tokens setting
            metadata: Any additional metadata for this interaction
        """
        self.logger.info(f"Model input: {prompt[:100]}...")
        
        data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **(metadata or {})
        }
        
        self._log_to_json("model_input", "Model input", data)
    
    def log_model_output(
        self, 
        response: str, 
        elapsed_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log output from a language model.
        
        Args:
            response: The response text from the model
            elapsed_time: Time taken to generate the response (in seconds)
            metadata: Any additional metadata for this interaction
        """
        self.logger.info(f"Model output: {response[:100]}...")
        
        data = {
            "response": response,
            "elapsed_time": elapsed_time,
            **(metadata or {})
        }
        
        self._log_to_json("model_output", "Model output", data)
    
    def log_reasoning_step(
        self, 
        step_number: int, 
        step_name: str, 
        input_data: Any, 
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a reasoning step in the template-oriented prompting process.
        
        Args:
            step_number: Number of the reasoning step (for ordering)
            step_name: Name of the reasoning step
            input_data: Input data for this step
            output_data: Output data from this step
            metadata: Any additional metadata for this step
        """
        self.logger.info(f"Reasoning step {step_number}: {step_name}")
        
        # Handle non-serializable objects
        try:
            input_str = json.dumps(input_data)
        except (TypeError, OverflowError):
            input_str = str(input_data)
        
        try:
            output_str = json.dumps(output_data)
        except (TypeError, OverflowError):
            output_str = str(output_data)
        
        log_data = {
            "step_number": step_number,
            "step_name": step_name,
            "input": input_data if isinstance(input_data, (str, int, float, bool, type(None))) else input_str,
            "output": output_data if isinstance(output_data, (str, int, float, bool, type(None))) else output_str,
            **(metadata or {})
        }
        
        self._log_to_json("reasoning_step", f"Reasoning step {step_number}: {step_name}", log_data)
    
    def log_template_operation(
        self, 
        operation: str, 
        template_id: Optional[str] = None,
        template_content: Optional[str] = None,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an operation on a template.
        
        Args:
            operation: Type of operation (e.g., 'retrieve', 'instantiate', 'update')
            template_id: ID of the template
            template_content: Content of the template
            success: Whether the operation was successful
            metadata: Any additional metadata for this operation
        """
        self.logger.info(f"Template operation: {operation} {template_id or ''}")
        
        data = {
            "operation": operation,
            "template_id": template_id,
            "template_content": template_content,
            "success": success,
            **(metadata or {})
        }
        
        self._log_to_json("template_operation", f"Template operation: {operation}", data)
    
    def log_error_detection(
        self, 
        error_type: str, 
        error_location: Optional[str] = None,
        error_description: str = "",
        solution: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error detected during the reasoning process.
        
        Args:
            error_type: Type of error detected
            error_location: Location of the error in the solution
            error_description: Description of the error
            solution: The solution that contains the error
            metadata: Any additional metadata for this error
        """
        self.logger.warning(f"Error detected: {error_type} - {error_description}")
        
        data = {
            "error_type": error_type,
            "error_location": error_location,
            "error_description": error_description,
            "solution": solution,
            **(metadata or {})
        }
        
        self._log_to_json("error_detection", f"Error detected: {error_type}", data)
    
    def log_experiment_start(
        self, 
        experiment_name: str, 
        config: Dict[str, Any]
    ) -> None:
        """
        Log the start of an experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration parameters for the experiment
        """
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.set_context(experiment_name=experiment_name)
        
        data = {
            "experiment_name": experiment_name,
            "config": config,
            "start_time": datetime.now().isoformat()
        }
        
        self._log_to_json("experiment_start", f"Starting experiment: {experiment_name}", data)
    
    def log_experiment_end(
        self, 
        experiment_name: str, 
        metrics: Dict[str, Any]
    ) -> None:
        """
        Log the end of an experiment.
        
        Args:
            experiment_name: Name of the experiment
            metrics: Metrics and results from the experiment
        """
        self.logger.info(f"Experiment completed: {experiment_name}")
        
        data = {
            "experiment_name": experiment_name,
            "metrics": metrics,
            "end_time": datetime.now().isoformat()
        }
        
        self._log_to_json("experiment_end", f"Experiment completed: {experiment_name}", data)
    
    def create_timed_section(self, section_name: str) -> 'TimedSection':
        """
        Create a timed section context manager.
        
        Args:
            section_name: Name of the section to time
            
        Returns:
            A context manager that times the section
        """
        return TimedSection(self, section_name)
    
    def save_metrics(self, metrics: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            filename: Optional filename (default: based on experiment_id)
        """
        if filename is None:
            filename = f"metrics_{self.experiment_id}.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Saved metrics to {filepath}")


class TimedSection:
    """Context manager for timing sections of code with automatic logging.
    
    Supports both synchronous and asynchronous context manager protocols.
    Can be used with both 'with' and 'async with' statements.
    """
    
    def __init__(self, logger: ToP_Logger, section_name: str):
        """
        Initialize a timed section.
        
        Args:
            logger: The ToP_Logger instance
            section_name: Name of the section to time
        """
        self.logger = logger
        self.section_name = section_name
        self.start_time = None
    
    def __enter__(self):
        """Start timing when entering the context (synchronous)."""
        self.start_time = time.time()
        self.logger.info(f"Starting section: {self.section_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log timing when exiting the context (synchronous)."""
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.logger.info(
                f"Completed section: {self.section_name} in {elapsed_time:.2f}s",
                section=self.section_name,
                elapsed_time=elapsed_time
            )
    
    async def __aenter__(self):
        """Start timing when entering the context (asynchronous)."""
        self.start_time = time.time()
        self.logger.info(f"Starting async section: {self.section_name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Log timing when exiting the context (asynchronous)."""
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.logger.info(
                f"Completed async section: {self.section_name} in {elapsed_time:.2f}s",
                section=self.section_name,
                elapsed_time=elapsed_time
            )


# Create a default singleton instance for easy import
default_logger = ToP_Logger()

# Add a diagnostic log to help debug logger initialization
default_logger.info("Default ToP_Logger initialized successfully")
default_logger.debug("TimedSection now supports both sync and async context manager protocols")

# Convenience functions to use the default logger
def get_logger(name: str = "top_logger", **kwargs) -> ToP_Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        **kwargs: Additional parameters to pass to ToP_Logger constructor
        
    Returns:
        A ToP_Logger instance
    """
    return ToP_Logger(name=name, **kwargs)

def set_default_logger(logger: ToP_Logger) -> None:
    """
    Set the default logger instance.
    
    Args:
        logger: The logger instance to set as default
    """
    global default_logger
    default_logger = logger
