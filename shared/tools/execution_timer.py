import atexit
import time

# Record start time immediately when this module is imported
_START_TIME = time.perf_counter()

def _format_duration(seconds: float) -> str:
    """Formats seconds into a clean, readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"


def _report_uptime() -> None:
    """Automatically prints the total run time on program exit."""
    duration = time.perf_counter() - _START_TIME
    print(f"\n[Execution Timer] Total runtime: {_format_duration(duration)}")


# Register the function to run automatically when the Python interpreter exits
atexit.register(_report_uptime)