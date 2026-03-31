try:
    from .ci_eval import CIEvaluator
except ImportError:
    CIEvaluator = None  # openai not installed

__all__ = ["CIEvaluator"]
