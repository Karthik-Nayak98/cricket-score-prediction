import logging

def setup_logging():
    """Initializes and returns the root or specialized logger configuration."""
    logger = logging.getLogger("cricket_pipeline")
    
    # If handlers are already configured, don't add them again (prevents duplicate logs)
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.DEBUG)

    # Handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_building_errors.log")
    file_handler.setLevel(logging.ERROR)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger