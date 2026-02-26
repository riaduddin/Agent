"""
Debug Logger Utility
====================
A configurable logging utility that can be toggled on/off via config.

Usage:
    from app.utils.debug_logger import debug_log, debug_warn, debug_error, debug_perf

    debug_log("My debug message")
    debug_log("Formatted message", value=123)
    debug_warn("Warning message")
    debug_error("Error message")
    debug_perf("PERF_LOG: Operation", time_taken=1.234)

Pattern-Based Filtering:
    Use set_debug_patterns() to enable logging only for specific prefixes:
    
    set_debug_patterns(["[EXTRACTION]", "[WORKER]"])
    debug_log("[EXTRACTION] Processing...")  # This WILL print
    debug_log("[CHAT] Sending message...")   # This will NOT print
    debug_log("General message...")          # This will NOT print
    
    set_debug_patterns([])  # Clear patterns to log everything again

Configuration:
    Set DEBUG_MODE = True in config.py to enable debug logging.
    Set DEBUG_MODE = False to disable all debug output.
    Set DEBUG_PATTERNS in config.py or env var (comma-separated) for pattern filtering.
"""

import os
from functools import wraps
from datetime import datetime

# Module-level cache for DEBUG_MODE to avoid circular imports
_debug_mode_cached = None

# Module-level pattern filter - only log messages starting with these patterns
# If empty, all debug messages are logged (default behavior)
_debug_patterns = []

# ANSI Color Codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _get_debug_mode() -> bool:
    """
    Get DEBUG_MODE from config, with caching to avoid repeated imports.
    Returns False if config cannot be loaded.
    """
    global _debug_mode_cached
    
    if _debug_mode_cached is not None:
        return _debug_mode_cached
    
    try:
        from app import config
        _debug_mode_cached = getattr(config, 'DEBUG_MODE', False)
    except ImportError:
        # Fallback: check environment variable directly
        _debug_mode_cached = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    return _debug_mode_cached


def reset_debug_mode_cache():
    """
    Reset the cached DEBUG_MODE value.
    Useful for testing or when config changes at runtime.
    """
    global _debug_mode_cached
    _debug_mode_cached = None


def set_debug_patterns(patterns: list):
    """
    Set the debug patterns filter. Only messages starting with these patterns will be logged.
    
    Args:
        patterns: List of string patterns to filter by (e.g., ["[EXTRACTION]", "[WORKER]"])
                  Pass empty list [] to disable filtering and log all messages.
    
    Examples:
        set_debug_patterns(["[EXTRACTION]"])  # Only log extraction-related messages
        set_debug_patterns(["[WORKER]", "[OCR]"])  # Log worker and OCR messages
        set_debug_patterns([])  # Log everything (default behavior)
    """
    global _debug_patterns
    _debug_patterns = patterns


def get_debug_patterns() -> list:
    """
    Get the current debug patterns filter.
    
    Returns:
        List of active pattern filters
    """
    return _debug_patterns.copy()


def _matches_debug_pattern(message: str) -> bool:
    """
    Check if the message matches any of the configured debug patterns.
    If no patterns are set, always returns True (log everything).
    
    Args:
        message: The message to check
        
    Returns:
        True if message should be logged, False otherwise
    """
    # If no patterns configured, log everything
    if not _debug_patterns:
        return True
    
    # Check if message starts with any of the patterns
    return any(message.startswith(pattern) for pattern in _debug_patterns)


def _load_patterns_from_config():
    """
    Load debug patterns from config or environment variable.
    Called once on first debug_log call.
    """
    global _debug_patterns
    
    try:
        from app import config
        patterns = getattr(config, 'DEBUG_PATTERNS', None)
        if patterns:
            _debug_patterns = patterns if isinstance(patterns, list) else [patterns]
            return
    except ImportError:
        pass
    
    # Fallback: check environment variable (comma-separated)
    env_patterns = os.getenv('DEBUG_PATTERNS', '')
    if env_patterns:
        _debug_patterns = [p.strip() for p in env_patterns.split(',') if p.strip()]


def debug_log(*args, **kwargs):
    """
    Print debug message if DEBUG_MODE is enabled and message matches pattern filter.
    
    Args:
        *args: Positional arguments to print
        **kwargs: Keyword arguments formatted as key=value pairs
    
    Pattern Filtering:
        If patterns are set via set_debug_patterns(), only messages starting with
        those patterns will be logged. Use prefixes like "[EXTRACTION]", "[WORKER]", etc.
    
    Examples:
        debug_log("Processing document", doc_id=123)
        debug_log("[EXTRACTION] Entity extraction started", doc_id=456)
        debug_log("[WORKER] Task completed", task_id="abc")
    """
    if not _get_debug_mode():
        return
    
    message_parts = [str(arg) for arg in args]
    
    if kwargs:
        kv_parts = [f"{k}={v}" for k, v in kwargs.items()]
        message_parts.extend(kv_parts)
    
    full_message = ' '.join(message_parts)
    
    # Check if message matches any configured pattern
    if not _matches_debug_pattern(full_message):
        return
    
    print(f"[DEBUG] {full_message}")




def debug_warn(*args, **kwargs):
    """
    Print warning message if DEBUG_MODE is enabled.
    
    Args:
        *args: Positional arguments to print
        **kwargs: Keyword arguments formatted as key=value pairs
    """
    if not _get_debug_mode():
        return
    
    message_parts = [str(arg) for arg in args]
    
    if kwargs:
        kv_parts = [f"{k}={v}" for k, v in kwargs.items()]
        message_parts.extend(kv_parts)
    
    print(f"[WARN] {' '.join(message_parts)}")


def debug_error(*args, **kwargs):
    """
    Print error message if DEBUG_MODE is enabled.
    Note: Consider using proper logging for production errors.
    
    Args:
        *args: Positional arguments to print
        **kwargs: Keyword arguments formatted as key=value pairs
    """
    if not _get_debug_mode():
        return
    
    message_parts = [str(arg) for arg in args]
    
    if kwargs:
        kv_parts = [f"{k}={v}" for k, v in kwargs.items()]
        message_parts.extend(kv_parts)
    
    print(f"{Colors.FAIL}[ERROR] {' '.join(message_parts)}{Colors.ENDC}")


def debug_perf(operation: str, time_taken: float = None, **kwargs):
    """
    Print performance log if DEBUG_MODE is enabled.
    
    Args:
        operation: Description of the operation being measured
        time_taken: Time in seconds (optional)
        **kwargs: Additional key=value pairs to log
    
    Examples:
        debug_perf("Query Embedding", time_taken=0.5)
        debug_perf("Vector Search", time_taken=1.2, neighbors=50)
    """
    if not _get_debug_mode():
        return
    
    parts = [f"[PERF] {operation}"]
    
    if time_taken is not None:
        parts.append(f"took {time_taken:.2f}s")
    
    if kwargs:
        kv_parts = [f"{k}={v}" for k, v in kwargs.items()]
        parts.append(f"({', '.join(kv_parts)})")
    
    print(' '.join(parts))


def debug_separator(title: str = None):
    """
    Print a visual separator line if DEBUG_MODE is enabled.
    
    Args:
        title: Optional title to display in the separator
    """
    if not _get_debug_mode():
        return
    
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print(f"{'='*50}")


def debug_banner(title: str, content: dict = None):
    """
    Print a formatted banner with optional content if DEBUG_MODE is enabled.
    
    Args:
        title: Banner title
        content: Optional dictionary of key-value pairs to display
    """
    if not _get_debug_mode():
        return
    
    border = "+" + "-" * 58 + "+"
    print(border)
    print(f"| {title.center(56)} |")
    print(border)
    
    if content:
        for key, value in content.items():
            line = f"{key}: {value}"
            if len(line) > 56:
                line = line[:53] + "..."
            print(f"| {line.ljust(56)} |")
        print(border)


def debug_trigger(action: str, details: str = None, source: str = "PUB/SUB"):
    """
    Print a professional formatted log for external triggers.
    Format: [SOURCE] → ACTION: Details
    """
    if not _get_debug_mode():
        return
    
    prefix = f"[{source.upper()}]"
    arrow = "→"
    msg = f"{prefix} {arrow} {action.upper()}"
    if details:
        msg = f"{msg}: {details}"
    
    # Use cyan/blue markers in terminal if supported
    print(f"\n{Colors.CYAN}{msg}{Colors.ENDC}")


def debug_connection(service: str, status: str, details: str = None):
    """
    Print a professional connection status log.
    Format: [CONNECTION] → SERVICE is STATUS (details)
    """
    if not _get_debug_mode():
        return
    
    prefix = "[CONNECTION]"
    msg = f"{prefix} → {service} is {status.upper()}"
    if details:
        msg = f"{msg} ({details})"
    
    print(f"{Colors.BLUE}{msg}{Colors.ENDC}")


def debug_warning(source: str, details: str):
    """
    Print a warning log with visual indicator.
    Format: [WARNING] ⚠ SOURCE: Details
    """
    if not _get_debug_mode():
        return
    
    prefix = "[WARNING]"
    warn = "⚠" # Unicode warning sign
    msg = f"{prefix} {warn} {source.upper()}: {details}"
    
    # Use yellow for warnings
    print(f"{Colors.WARNING}{msg}{Colors.ENDC}")


def debug_success(topic: str, details: str = None):
    """
    Print a success log with visual indicator.
    Format: [SUCCESS] ✓ TOPIC: Details
    """
    if not _get_debug_mode():
        return
    
    prefix = "[SUCCESS]"
    check = "✓"
    msg = f"{prefix} {check} {topic.upper()}"
    if details:
        msg = f"{msg}: {details}"
    
    print(f"{Colors.GREEN}{msg}{Colors.ENDC}")
