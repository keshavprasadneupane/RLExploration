from collections.abc import Callable
from functools import wraps
from typing import Any


def stop_by_user(
	message: str = "Operation interrupted by user",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
	def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
		@wraps(func)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			try:
				return func(*args, **kwargs)
			except KeyboardInterrupt:
				print(f"\n{message}")
				return None

		return wrapper

	return decorator