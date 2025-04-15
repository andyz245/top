"""
Test script to validate the ToP_Logger implementation.
"""

import logging
import asyncio
from logger import ToP_Logger

async def test_asyncio_debug():
    """Test that asyncio debug mode works with our logger."""
    logger = ToP_Logger(name="test_asyncio")
    
    # This would fail if isEnabledFor is not implemented
    asyncio.get_event_loop().set_debug(True)
    
    # Test standard logging methods
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Test with standard logging kwargs
    logger.error("Error with exception info", exc_info=True)
    
    # Test level check
    print(f"Is DEBUG enabled: {logger.isEnabledFor(logging.DEBUG)}")
    
    # Test custom data
    logger.info("Message with custom data", custom_field="test value")
    
    return "All tests passed!"

if __name__ == "__main__":
    result = asyncio.run(test_asyncio_debug())
    print(result)
