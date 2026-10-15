from functools import wraps
from typing import Any, Callable, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


@overload
def stop_by_user(
	func: None = None,
	post_exception: Callable[[], None] | None = None,
	*,
	message: str = "Operation interrupted by user",
) -> Callable[[F], F]: ...


@overload
def stop_by_user(
	func: F, # pyright: ignore[reportInvalidTypeVarUse]
	post_exception: Callable[[], None] | None = None,
	*,
	message: str = "Operation interrupted by user",
	**kwargs: Any,
) -> Any: ...


def stop_by_user(
	func: F | None = None, # pyright: ignore[reportInvalidTypeVarUse]
	post_exception: Callable[[], None] | None = None,
	*,
	message: str = "Operation interrupted by user",
	**kwargs: Any,
) -> Any:
	"""
	Catch KeyboardInterrupt during function execution.

	Direct execution:
		stop_by_user(train, post_exception=cleanup, value=10)

	Decorator:
		@stop_by_user(message="Interrupted")
		def train():
			...
	"""

	def decorator(target: F) -> F:
		@wraps(target)
		def wrapper(*args: Any, **call_kwargs: Any) -> Any:
			try:
				return target(*args, **call_kwargs)
			except KeyboardInterrupt:
				print(f"\\n{message}")
				if post_exception is not None:
					post_exception()
				return None

		return wrapper  # type: ignore[return-value]

	if func is None:
		return decorator

	return decorator(func)(**kwargs)