"""Domain exceptions. Mapped to HTTP in interfaces only."""


class DomainError(Exception):
    """Base error for the receipt-extraction context."""


class PasswordProtected(DomainError):
    """PDF requires a password we do not have."""


class CorruptDocument(DomainError):
    """PDF is truncated, malformed, or unreadable."""


class TooManyPages(DomainError):
    def __init__(self, page_count: int, max_pages: int) -> None:
        self.page_count = page_count
        self.max_pages = max_pages
        super().__init__(f"PDF has {page_count} pages; max is {max_pages}")


class FileTooLarge(DomainError):
    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(f"File is {size_bytes} bytes; max is {max_bytes}")


class UnsupportedDocument(DomainError):
    def __init__(self, content_type: str) -> None:
        self.content_type = content_type
        super().__init__(f"Unsupported content type: {content_type}")


class InferenceTimeout(DomainError):
    """A page exceeded INFERENCE_TIMEOUT_S."""


class GpuBusy(DomainError):
    """Waiter exceeded QUEUE_TIMEOUT_S (HTTP 503)."""


class QueueFull(DomainError):
    """In-process GPU queue is at capacity (HTTP 429)."""
