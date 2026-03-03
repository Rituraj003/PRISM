"""
Unified answer verification and parsing for both multiple-choice and math problems.

This module provides core utility functions for extracting and verifying answers consistently
across all methods, handling various output formats including <answer> tags, \\boxed{} notation, etc.

Core utility functions:
- extract_mcq_answer(): Extract multiple choice answer (a-d) from model response
- extract_math_answer(): Extract math answer from model response
- normalize_answer(): Normalize an answer for comparison
- check_answer_correctness(): Check if predicted answer matches correct answer (the main "is_correct" function)
- is_numerical_match(): Robust numerical matching with tolerance
- clean_answer(): Clean and normalize answer strings
- parse_answer_with_reasoning(): Parse answer from LLM response, separating reasoning from final answer
- parse_verifier_verdict(): Parse verifier output (PRM or ORM), extracting verdict
- extract_prm_step_scores(): Extract per-step scores from PRM verifier feedback
- compute_prm_score(): Compute a scalar PRM score in [0, 1] from verifier output

This module contains NO implementation logic (like majority voting, LLM aggregation, etc).
Methods decide their own aggregation/verification strategies.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from typing import TYPE_CHECKING, Any, cast

from dateutil import parser
from sympy import simplify, sympify
from sympy.parsing.latex import parse_latex

from shared import QuestionType

if TYPE_CHECKING:
    from shared import EvaluationExample

# Multiple choice answer patterns - try in order of specificity
MCQ_ANSWER_PATTERNS = [
    # Standard format: <answer>a</answer>
    re.compile(r"<answer>\s*([a-dA-D])\s*</answer>"),
    # With LaTeX boxed: <answer>\boxed{a}</answer>
    re.compile(r"<answer>\s*\\boxed\{([a-dA-D])\}\s*</answer>"),
    # Boxed without answer tags: \boxed{a}
    re.compile(r"\\boxed\{([a-dA-D])\}"),
    # Boxed with \text tags: \boxed{\text{a}}
    re.compile(r"\\boxed\{\\text\{([a-dA-D])\}\}"),
    # Just the letter at end of response (fallback)
    re.compile(r"\b([a-dA-D])\s*[.)]?\s*$", re.MULTILINE),
]

# Math answer patterns
MATH_ANSWER_PATTERNS = [
    # Standard format: <answer>value</answer>
    re.compile(r"<answer>(.*?)</answer>", re.DOTALL),
    # With LaTeX boxed: \boxed{value} - handles nested braces by matching to last } before $$ or newline
    re.compile(r"\\boxed\{(.+?)\}\s*(?:\$\$|$)", re.DOTALL),
    # With LaTeX boxed (double braces): \boxed{{value}}
    re.compile(r"\\boxed\{\{(.+?)\}\}\s*(?:\$\$|$)", re.DOTALL),
]

# Text answer patterns
TEXT_ANSWER_PATTERNS = [
    # Standard format: <text>value</text>
    re.compile(r"\<text\>(.*?)\</text\>", re.DOTALL),
    # With LaTeX boxed: \boxed{value}
    re.compile(r"\\text\{([^}]+)\}"),
    # With LaTeX boxed: \boxed{value}
    re.compile(r"\\boxed\{([^}]+)\}"),
    # With LaTeX boxed: \boxed{value}
    re.compile(r"\\boxed\{\\text\{([^}]+)\}\}"),
]

# Unicode normalization table for sign variants (minus/plus)
# Some models emit U+2212 (minus sign) or other dash characters which break parsing
_UNICODE_SIGN_TABLE = {
    ord("\u2212"): "-",  # minus sign
    ord("\u2010"): "-",  # hyphen
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u2012"): "-",  # figure dash
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\uFE63"): "-",  # small hyphen-minus
    ord("\uFF0D"): "-",  # fullwidth hyphen-minus
    ord("\uFF0B"): "+",  # fullwidth plus
}


# LaTeX normalization patterns
SQRT_SIMPLE_RE = re.compile(r"\\sqrt\s*([0-9]+(?:\.[0-9]+)?)")
SQRT_VAR_RE = re.compile(r"\\sqrt\s*([a-zA-Z])")
FRAC_NUM_BRACED_RE = re.compile(
    r"\\frac\s*\{([0-9]+(?:\.[0-9]+)?)\}\s*([0-9]+(?:\.[0-9]+)?)(?![0-9.])"
)
FRAC_DEN_BRACED_RE = re.compile(
    r"\\frac\s*([0-9]+(?:\.[0-9]+)?)\s*\{([0-9]+(?:\.[0-9]+)?)\}"
)
FRAC_SIMPLE_RE = re.compile(
    r"\\frac\s*([0-9]+(?:\.[0-9]+)?)\s*([0-9]+(?:\.[0-9]+)?)(?![0-9.])"
)
LATEX_SPACING_RE = re.compile(r"\\(?:,|;|:|!|quad|qquad)")
LATEX_WS_RE = re.compile(r"\\\s*")
TEXT_CMD_RE = re.compile(r"\\text\{[^}]*\}")
DEGREE_RE = re.compile(r"\^\{\\circ\}")
SIMPLE_LHS_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*$|^\\[A-Za-z]+(?:_{[^}]+}|_[A-Za-z0-9+-]+)?$"
)

LOG10_2 = math.log10(2)


def _extract_braced(expr: str, start: int) -> tuple[str | None, int]:
    if start >= len(expr) or expr[start] != "{":
        return None, start
    depth = 0
    for i in range(start, len(expr)):
        if expr[i] == "{":
            depth += 1
        elif expr[i] == "}":
            depth -= 1
            if depth == 0:
                return expr[start + 1 : i], i + 1
    return None, start


def _replace_frac(expr: str) -> str:
    result = []
    i = 0
    while i < len(expr):
        if expr.startswith(r"\frac{", i):
            i += len(r"\frac")
            numerator, i = _extract_braced(expr, i)
            if numerator is None:
                result.append(r"\frac")
                continue
            denominator, i = _extract_braced(expr, i)
            if denominator is None:
                result.append(f"({numerator})")
                continue
            result.append(f"({numerator})/({denominator})")
            continue
        result.append(expr[i])
        i += 1
    return "".join(result)


def _replace_sqrt(expr: str) -> str:
    result = []
    i = 0
    while i < len(expr):
        if expr.startswith(r"\sqrt{", i):
            i += len(r"\sqrt")
            value, i = _extract_braced(expr, i)
            if value is None:
                result.append(r"\sqrt")
                continue
            result.append(f"sqrt({value})")
            continue
        result.append(expr[i])
        i += 1
    return "".join(result)


def evaluate_latex_expression(expr: str) -> float | None:
    """
    Try to evaluate a LaTeX mathematical expression to a numerical value using sympy.

    Args:
        expr: LaTeX expression string

    Returns:
        Numerical value or None if evaluation fails
    """
    if not expr:
        return None

    try:
        # Handle escaped backslashes from CSV
        expr = expr.replace("\\\\", "\\")

        # Use sympy's LaTeX parser
        if "!" in expr:
            raise ValueError("factorial")
        result = parse_latex(expr)
        if result is None:
            raise ValueError("parse_latex returned None")

        if bool(getattr(result, "is_number", False)):
            return float(cast(Any, result))
        evalf = getattr(result, "evalf", None)
        if callable(evalf):
            evaled = evalf()
            if bool(getattr(evaled, "is_number", False)):
                return float(cast(Any, evaled))

    except Exception:
        pass

    try:
        sympy_expr = _latex_to_sympy_expr(expr)
        result = sympify(sympy_expr)
        if bool(getattr(result, "is_number", False)):
            return float(cast(Any, result))
        evalf = getattr(result, "evalf", None)
        if callable(evalf):
            evaled = evalf()
            if bool(getattr(evaled, "is_number", False)):
                return float(cast(Any, evaled))
    except Exception:
        return None

    return None


def evaluate_latex_expression_sympy(expr: str):
    """
    Try to parse a math expression into a sympy expression (without float evaluation).

    This is useful for canonicalizing exact rationals (e.g. "\\frac{1}{576}" -> 1/576)
    without introducing floating-point artifacts.
    """
    if not expr:
        return None

    try:
        expr = expr.replace("\\\\", "\\")
        if "!" in expr:
            raise ValueError("factorial")
        return parse_latex(expr)
    except Exception:
        pass

    try:
        sympy_expr = _latex_to_sympy_expr(expr)
        return sympify(sympy_expr)
    except Exception:
        return None


def normalize_latex(expr: str) -> str:
    """
    Normalize LaTeX expressions for better comparison.
    """
    if not expr:
        return expr

    # Replace \dfrac/\tfrac with \frac
    expr = expr.replace(r"\dfrac", r"\frac")
    expr = expr.replace(r"\tfrac", r"\frac")
    # Normalize common missing-brace patterns
    expr = FRAC_NUM_BRACED_RE.sub(r"\\frac{\1}{\2}", expr)
    expr = FRAC_DEN_BRACED_RE.sub(r"\\frac{\1}{\2}", expr)
    expr = FRAC_SIMPLE_RE.sub(r"\\frac{\1}{\2}", expr)
    expr = SQRT_SIMPLE_RE.sub(r"\\sqrt{\1}", expr)
    expr = SQRT_VAR_RE.sub(r"\\sqrt{\1}", expr)
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = TEXT_CMD_RE.sub("", expr)
    expr = LATEX_SPACING_RE.sub("", expr)
    expr = expr.replace(r"\approx", "").replace(r"\sim", "")
    # Remove extra spaces
    expr = re.sub(r"\s+", "", expr)
    return expr


def _split_top_level_commas(expr: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in expr:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _looks_like_simple_lhs(expr: str) -> bool:
    candidate = re.sub(r"\s+", "", expr)
    if not candidate:
        return False
    if re.search(r"\d", candidate):
        return False
    if any(op in candidate for op in "+-*/^"):
        return False
    return SIMPLE_LHS_RE.fullmatch(candidate) is not None


def _strip_assignment(expr: str) -> str:
    if expr.count("=") != 1:
        return expr
    left, right = expr.split("=", 1)
    if _looks_like_simple_lhs(left) and right.strip():
        return right
    return expr


def _has_numeric_token(expr: str) -> bool:
    return bool(re.search(r"\d", expr) or r"\frac" in expr or r"\sqrt" in expr)


def _strip_approx(expr: str) -> str:
    for token in (r"\approx", r"\sim"):
        if token in expr:
            left, right = expr.split(token, 1)
            left = left.strip()
            right = right.strip()
            if _has_numeric_token(left):
                expr = left
            elif _has_numeric_token(right):
                expr = right
            else:
                expr = left or right
    return expr


def _int_digits_exceed_limit(value: int, limit: int) -> bool:
    if limit <= 0:
        return False
    value = abs(value)
    if value == 0:
        return False
    est_digits = int(value.bit_length() * LOG10_2) + 1
    return est_digits >= limit


def _int_to_bytes(value: int) -> bytes:
    if value == 0:
        return b"\x00"
    length = (value.bit_length() + 7) // 8
    try:
        return value.to_bytes(length, "big", signed=value < 0)
    except OverflowError:
        return value.to_bytes(length + 1, "big", signed=True)


def _hash_large_rational(numerator: int, denominator: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"rational:v1:")
    hasher.update(_int_to_bytes(numerator))
    hasher.update(b"/")
    hasher.update(_int_to_bytes(denominator))
    return f"rational:sha256:{hasher.hexdigest()}"


def _safe_rational_str(sympy_val) -> str:
    limit_getter = getattr(sys, "get_int_max_str_digits", None)
    limit = limit_getter() if limit_getter is not None else 0
    numerator = int(sympy_val.p)
    denominator = int(sympy_val.q)
    if limit and (
        _int_digits_exceed_limit(numerator, limit)
        or _int_digits_exceed_limit(denominator, limit)
    ):
        return _hash_large_rational(numerator, denominator)
    try:
        return str(sympy_val)
    except ValueError:
        return _hash_large_rational(numerator, denominator)


def _normalize_math_atom(ans: str) -> str:
    ans = ans.replace("$", "")
    ans = clean_answer(ans)
    ans = _strip_assignment(ans)
    ans = _strip_approx(ans)
    ans = normalize_latex(ans)

    try:
        value = float(ans)
        return str(int(value)) if value.is_integer() else str(value)
    except ValueError:
        pass

    # Prefer exact rational canonicalization when possible to avoid float artifacts.
    sympy_val = evaluate_latex_expression_sympy(ans)
    if sympy_val is not None:
        try:
            sympy_val = simplify(sympy_val)
        except Exception:
            pass
        try:
            if sympy_val.is_rational:
                return _safe_rational_str(sympy_val)
        except AttributeError:
            # sympy_val might be the singleton registry S, not an expression
            pass

    val = evaluate_latex_expression(ans)
    if val is not None:
        return str(int(val)) if val.is_integer() else str(val)

    return ans.lower().replace(" ", "")


def _latex_to_sympy_expr(expr: str) -> str:
    """
    Convert a subset of LaTeX to a sympy-friendly expression string.
    """
    expr = normalize_latex(expr)
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = expr.replace(r"\cdot", "*").replace(r"\times", "*")
    expr = expr.replace(r"\pi", "pi")
    expr = (
        expr.replace(r"\,", "").replace(r"\;", "").replace(r"\!", "").replace(r"\:", "")
    )
    expr = expr.replace(r"\approx", "").replace(r"\sim", "")
    expr = expr.replace(r"\displaystyle", "")
    expr = TEXT_CMD_RE.sub("", expr)
    expr = DEGREE_RE.sub("", expr)
    expr = expr.replace(r"\circ", "")

    expr = _replace_frac(expr)
    expr = _replace_sqrt(expr)

    expr = LATEX_WS_RE.sub("", expr)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("^", "**")

    expr = re.sub(r"(\d+)!", r"factorial(\1)", expr)
    expr = re.sub(r"([a-zA-Z])!", r"factorial(\1)", expr)

    expr = re.sub(r"(\d)(pi)", r"\1*pi", expr)
    expr = re.sub(r"(\d)(sqrt\()", r"\1*sqrt(", expr)
    expr = re.sub(r"\)(?=\()", r")*(", expr)
    expr = re.sub(r"(\d)(\()", r"\1*(", expr)
    expr = re.sub(r"(\))(\d)", r"\1*\2", expr)
    expr = re.sub(r"(\))([a-zA-Z])", r"\1*\2", expr)

    return expr


def is_numerical_match(answer1: str, answer2: str, tolerance: float = 1e-6) -> bool:
    """
    Check if two answers match, either exactly or numerically within tolerance.

    Args:
        answer1: First answer string
        answer2: Second answer string
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if answers match
    """
    # First normalize both answers to extract from \boxed{} and parse
    answer1_norm = normalize_math_answer(answer1)
    answer2_norm = normalize_math_answer(answer2)

    # Try exact string match on normalized
    if answer1_norm == answer2_norm:
        return True

    list1 = _split_top_level_commas(answer1_norm)
    list2 = _split_top_level_commas(answer2_norm)
    if len(list1) > 1 or len(list2) > 1:
        # Both must be lists of the same arity to match.
        if len(list1) != len(list2):
            return False

        def numeric_list(values: list[str]) -> list[float] | None:
            parsed = []
            for item in values:
                try:
                    parsed.append(float(item))
                    continue
                except ValueError:
                    pass
                val = evaluate_latex_expression(item)
                if val is None:
                    return None
                parsed.append(val)
            return parsed

        nums1 = numeric_list(list1)
        nums2 = numeric_list(list2)
        if nums1 is not None and nums2 is not None:
            nums1_sorted = sorted(nums1)
            nums2_sorted = sorted(nums2)
            return all(
                abs(a - b) < tolerance for a, b in zip(nums1_sorted, nums2_sorted)
            )

        list1_sorted = sorted(list1)
        list2_sorted = sorted(list2)
        return list1_sorted == list2_sorted

    # Try numerical comparison (direct float or evaluable expression)
    val1: float | None
    val2: float | None
    try:
        val1 = float(answer1_norm)
    except ValueError:
        val1 = evaluate_latex_expression(answer1_norm)
    try:
        val2 = float(answer2_norm)
    except ValueError:
        val2 = evaluate_latex_expression(answer2_norm)
    if val1 is not None and val2 is not None:
        return abs(val1 - val2) < tolerance

    # Try symbolic equivalence for algebraic expressions
    try:
        expr1 = sympify(_latex_to_sympy_expr(answer1_norm or answer1))
        expr2 = sympify(_latex_to_sympy_expr(answer2_norm or answer2))
        return simplify(cast(Any, expr1) - cast(Any, expr2)) == 0
    except Exception:
        return False

    # If we got here, they don't match
    return False


def clean_answer(answer: str) -> str:
    """
    Clean up the answer by removing extra whitespace and normalizing.
    Also handles incomplete LaTeX expressions.

    Args:
        answer: Raw answer string

    Returns:
        Cleaned answer string
    """
    if not answer:
        return ""

    # Remove extra whitespace
    answer = re.sub(r"\s+", " ", answer.strip())

    # Remove trailing periods and commas that might be artifacts
    answer = answer.rstrip(".,")

    # Remove incomplete LaTeX commands at the end
    answer = re.sub(r"\\[a-zA-Z]+\s*$", "", answer)

    # Remove unmatched opening braces at the end (more carefully)
    # Only remove if there's an opening brace without a corresponding closing brace
    open_braces = answer.count("{")
    close_braces = answer.count("}")
    if open_braces > close_braces:
        # Remove from the last unmatched opening brace to the end
        last_open = answer.rfind("{")
        if last_open != -1:
            answer = answer[:last_open]

    # Aggressively strip trailing non-alphanumeric chars (except closing braces/parens or factorial)
    # This handles "336^", "336$", "336-", etc.
    answer = re.sub(r"[^\w})!]+$", "", answer)

    return answer


def extract_mcq_answer(response: str) -> str | None:
    """
    Extract multiple choice answer (a, b, c, or d) from response.

    Tries multiple patterns in order of specificity to handle various formats.
    Returns the extracted letter in lowercase, or None if not found.

    Examples:
        >>> extract_mcq_answer("Some reasoning\n<answer>a</answer>")
        'a'
        >>> extract_mcq_answer("The answer is \\boxed{b}")
        'b'
        >>> extract_mcq_answer("<answer>\\boxed{C}</answer>")
        'c'
    """
    if not response:
        return None

    for pattern in MCQ_ANSWER_PATTERNS:
        match = pattern.search(response)
        if match:
            return match.group(1).lower()

    return None


def extract_math_answer(response: str) -> str | None:
    """
    Extract math answer from response.

    Tries multiple patterns to handle <answer> tags and \\boxed{} notation.
    For \\boxed{}, uses the LAST occurrence to avoid grabbing intermediate
    derivation steps (e.g., \\boxed{s_1} in the middle of a solution).
    Returns the raw extracted answer, or None if not found.

    Examples:
        >>> extract_math_answer("<answer>42</answer>")
        '42'
        >>> extract_math_answer("Therefore \\boxed{3.14}")
        '3.14'
    """
    if not response:
        return None

    # First try <answer> tags (use last match)
    matches = list(MATH_ANSWER_PATTERNS[0].finditer(response))
    if matches:
        answer = matches[-1].group(1).strip()
        if answer:
            return answer

    # Then try \boxed with balanced brace parsing - find the LAST occurrence
    # to avoid grabbing intermediate derivation boxes
    boxed_matches = list(re.finditer(r"\\boxed\{", response))
    if boxed_matches:
        # Use the last \boxed{} match
        boxed_match = boxed_matches[-1]
        start = boxed_match.end()
        count = 1
        end = start
        for i in range(start, len(response)):
            if response[i] == "{":
                count += 1
            elif response[i] == "}":
                count -= 1
                if count == 0:
                    end = i
                    break
        if end > start:
            content = response[start:end].strip()
            if content:
                return content

    return None


def extract_text_answer(response: str) -> str | None:
    """
    Extract text answer from response.

    Tries multiple patterns to handle <text> tags and \\boxed{} notation.
    Returns the raw extracted answer, or None if not found.

    Examples:
        >>> extract_text_answer("<text>Radcliffe College</text>")
        'Radcliffe College'
        >>> extract_text_answer("Therefore \\boxed{Radcliffe College}")
        'Radcliffe College'
    """
    if not response:
        return None

    for pattern in TEXT_ANSWER_PATTERNS:
        match = pattern.search(response)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer

    return None


def normalize_math_answer(ans: str | None) -> str:
    """
    Normalize a math answer for comparison.

    Handles LaTeX notation, fractions, square roots, and numeric values.
    Uses robust LaTeX parsing and numerical evaluation.
    Returns a normalized string representation.
    """
    if not ans:
        return ""

    # FIRST: Extract from \boxed{} or <answer> tags if present
    # This is critical for consistency between different answer formats
    extracted = extract_math_answer(ans)
    if extracted:
        ans = extracted

    ans = ans.replace("$", "")
    ans = clean_answer(ans)
    ans = _strip_assignment(ans)
    ans = _strip_approx(ans)
    ans = normalize_latex(ans)

    parts = _split_top_level_commas(ans)
    if len(parts) > 1:
        normalized_parts = [_normalize_math_atom(part) for part in parts]
        normalized_parts = sorted(normalized_parts)
        return ",".join(normalized_parts)

    return _normalize_math_atom(ans)


def parse_numerical_range(text):
    """
    Parses a string to extract the numerical range.

    1. Tries to find a range in parentheses (e.g., 11.41% and 11.65%).
    2. If no range is found, extracts the main value and returns it as a single-point range.
    """

    # Helper function to remove non-numeric characters (except dot) and convert to float
    def clean_and_convert(s):
        # Remove commas (for thousands), percent signs, and common currency symbols
        # The list of characters to remove has been slightly generalized
        cleaned = (
            s.replace(",", "")
            .replace("%", "")
            .replace("€", "")
            .replace("$", "")
            .replace("£", "")
            .strip()
        )
        return float(cleaned)

    # --- 1. Attempt to find the range in parentheses (Existing Logic) ---
    match_parentheses = re.search(r"\((.*?)\)", text)

    if match_parentheses:
        range_text = match_parentheses.group(1)
        # Use a flexible pattern to find numbers (digits, commas, dots) potentially followed by %
        float_strings = re.findall(
            r"[+-]?(?:\d[\d,]*\.?\d*|\.\d+)%?",
            range_text,
        )

        if len(float_strings) >= 2:
            try:
                lower_bound = clean_and_convert(float_strings[0])
                upper_bound = clean_and_convert(float_strings[1])
                return [lower_bound, upper_bound], None
            except ValueError:
                # Fall through to the single-value extraction if conversion fails
                pass

    # --- 1b. Attempt direct range forms (e.g., 12-14, -5--3) ---
    direct_range_match = re.match(
        r"^\s*([+-]?(?:\d[\d,]*\.?\d*|\.\d+))\s*-\s*([+-]?(?:\d[\d,]*\.?\d*|\.\d+))\s*$",
        text,
    )
    if direct_range_match:
        try:
            lower_bound = clean_and_convert(direct_range_match.group(1))
            upper_bound = clean_and_convert(direct_range_match.group(2))
            return [lower_bound, upper_bound], None
        except ValueError:
            pass

    # --- 2. If no valid range is found, extract the main value (New Logic) ---

    # *** REVISED REGEX HERE ***
    # Pattern now allows for optional currency symbols ($€£) at the beginning of the string
    # before the number sequence ([\d,.\s]+).
    main_value_match = re.search(
        r"^[\s]*([$€£]?[\s]*[+-]?[\d,.\s]+%?[\s]*\w*)",
        text,
    )

    if main_value_match:
        # Extract the matched group (e.g., "€120,000" or "12.27")
        main_value_string = main_value_match.group(1)

        # We search for the *actual* number part within this match, excluding trailing units
        # that might have been picked up by the \w*
        number_only_match = re.search(
            r"[$€£]?[\s]*[+-]?[\d,.]+%?",
            main_value_string,
        )

        if number_only_match:
            try:
                single_value = clean_and_convert(number_only_match.group(0))
                # Treat the single value as the min and max of the range
                return [
                    single_value,
                    single_value,
                ], "Note: Extracted single value as range."
            except ValueError:
                return None, "Error: Could not convert the main value to a number."

    return None, "Error: No recognizable range or single value found."


def normalize_text_answer(ans: str | None, question_type: QuestionType) -> str | None:
    """
    Normalize a text answer for comparison.

    Handles Person, Place, Other, Date, Number values.
    Use dateutil parser for Date.
    Returns a normalized string representation.
    """
    if not ans:
        return ""

    # Clean the answer first
    ans = clean_answer(ans)

    if question_type == QuestionType.DATE:
        # use dateutil parser to parse dates
        try:
            dt_object = parser.parse(ans)
            return dt_object.isoformat()
        except ValueError:
            pass

    elif question_type == QuestionType.NUMBER:
        # Extract numerical range
        numerical_range, note = parse_numerical_range(ans)
        if numerical_range is not None:
            return f"{numerical_range[0]}-{numerical_range[1]}"
        else:
            return None

    else:
        # Default text normalization
        pass

    # Fallback: remove spaces and lowercase
    return ans.lower().replace(" ", "")


def normalize_answer(answer: str | None, question_type: QuestionType) -> str | None:
    """
    Normalize an answer for comparison.

    Args:
        answer: The raw answer string
        question_type: The modality used for normalization

    Returns:
        Normalized answer string, or None if invalid

    Examples:
        >>> normalize_answer("A", question_type=QuestionType.MCQ)
        'a'
        >>> normalize_answer("42.0", question_type=QuestionType.MATH)
        '42'
    """
    if not answer:
        return None

    if question_type == QuestionType.MCQ:
        # For MCQ, just return the letter
        normalized = answer.strip().lower()
        if len(normalized) == 1 and normalized in "abcd":
            return normalized
        return None
    elif question_type == QuestionType.MATH:
        # For math, use the math normalization
        normalized = normalize_math_answer(answer)
        return normalized if normalized else None
    else:
        # For text, use the text normalization
        normalized = normalize_text_answer(answer, question_type)
        return normalized if normalized else None


_SIGNED_NUMBER_TOKEN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
_SIGNED_RANGE_RE = re.compile(
    rf"^\s*({_SIGNED_NUMBER_TOKEN})\s*(?:-\s*({_SIGNED_NUMBER_TOKEN}))?\s*$"
)


def _parse_numeric_range(value: str) -> tuple[float, float] | None:
    match = _SIGNED_RANGE_RE.fullmatch(value.strip())
    if not match:
        return None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) is not None else low
    return low, high


def compare(range_a: str | None, range_b: str | None) -> bool:
    """
    Compares two numeric ranges represented by list[float] parameters.

    Returns True if the numeric range indicated by the second parameter (range_b)
    is completely contained within the first parameter (range_a), otherwise False.

    A range is defined by the two values in the list (start and end).
    A single point is a valid range where both values are the same.

    Args:
        range_a: The containing range (e.g., "12.14-12.89").
        range_b: The contained range (e.g., "12.20-12.50").
    """
    # Add null check at the start
    if range_a is None or range_b is None:
        return False

    try:
        parsed_a = _parse_numeric_range(range_a)
        parsed_b = _parse_numeric_range(range_b)
        if parsed_a is None or parsed_b is None:
            return False
        a_min, a_max = parsed_a
        b_min, b_max = parsed_b

        a_low, a_high = (a_min, a_max) if a_min <= a_max else (a_max, a_min)
        b_low, b_high = (b_min, b_max) if b_min <= b_max else (b_max, b_min)

        # For range_b to be contained within range_a:
        # a) start(range_b) >= start(range_a)
        # b) end(range_b) <= end(range_a)
        is_contained = (a_low <= b_low) and (b_high <= a_high)

        return is_contained
    except (ValueError, IndexError):
        # If conversion fails, return False
        return False


def check_answer_correctness(
    predicted: str | None,
    correct: str | None,
    question_type: QuestionType,
    example: "EvaluationExample | None" = None,
) -> bool:
    """
    Check if a predicted answer matches the correct answer.

    This is the main "is_correct" function used across all methods.
    Uses robust numerical and LaTeX matching for math problems.

    Args:
        predicted: The predicted answer
        correct: The correct answer
        question_type: The modality governing comparison rules
        example: Optional EvaluationExample context (unused for current modalities)

    Returns:
        True if answers match, False otherwise

    Examples:
        >>> check_answer_correctness("a", "A", question_type=QuestionType.MCQ)
        True
        >>> check_answer_correctness("42.0", "42", question_type=QuestionType.MATH)
        True
        >>> check_answer_correctness("\\frac{1}{2}", "0.5", question_type=QuestionType.MATH)
        True
    """
    if predicted is None:
        return False

    if correct is None:
        return False

    if question_type == QuestionType.MCQ:
        # For MCQ, normalize and compare
        pred_norm = normalize_answer(predicted, question_type=QuestionType.MCQ)
        corr_norm = normalize_answer(correct, question_type=QuestionType.MCQ)
        return pred_norm == corr_norm
    elif question_type == QuestionType.MATH:
        # For math, use robust numerical matching
        return is_numerical_match(predicted, correct)
    elif question_type == QuestionType.NUMBER:
        # For text answers, clean and compare
        pred_norm = normalize_answer(predicted, question_type=QuestionType.NUMBER)
        corr_norm = normalize_answer(correct, question_type=QuestionType.NUMBER)
        return compare(corr_norm, pred_norm)
    else:
        # For text answers, clean and compare
        pred_norm = normalize_answer(predicted, question_type=question_type)
        corr_norm = normalize_answer(correct, question_type=question_type)
        return pred_norm == corr_norm


def _find_boxed_span(response: str) -> tuple[int, int] | None:
    """
    Find the start and end indices of a \\boxed{...} block using manual brace counting.
    This handles nested braces correctly, unlike regex.

    Returns:
        Tuple of (start, end) indices or None if not found.
    """
    boxed_match = re.search(r"\\boxed\{", response)
    if not boxed_match:
        return None

    start = boxed_match.start()
    brace_start = boxed_match.end()
    count = 1

    for i in range(brace_start, len(response)):
        if response[i] == "{":
            count += 1
        elif response[i] == "}":
            count -= 1
            if count == 0:
                return (start, i + 1)

    return None


def parse_answer_with_reasoning(
    response: str, question_type: QuestionType
) -> tuple[str, str | None]:
    """
    Parse answer from LLM response, separating reasoning from the final answer.

    Args:
        response: The LLM response text
        question_type: QuestionType describing extraction strategy

    Returns:
        A tuple of (reasoning, choice) where:
        - reasoning: The text before the answer tag
        - choice: The extracted answer (letter for MCQ, value for others)

    Examples:
        >>> parse_answer_with_reasoning("Let me think... <answer>a</answer>", question_type=QuestionType.MCQ)
        ('Let me think...', 'a')
        >>> parse_answer_with_reasoning("Solution: \\boxed{42}", question_type=QuestionType.MATH)
        ('Solution:', '42')
        >>> parse_answer_with_reasoning("Solution: <text>done</text>", question_type=QuestionType.OTHER)
        ('Solution:', 'done')
    """
    # Extract answer based on question type
    if question_type == QuestionType.MCQ:
        choice_str = extract_mcq_answer(response)
        tag_pattern = r"<answer>.*?</answer>"
    elif question_type.is_textual:
        choice_str = extract_text_answer(response)
        tag_pattern = r"<text>.*?</text>"
    else:
        choice_str = extract_math_answer(response)
        tag_pattern = r"<answer>.*?</answer>"

    # Find where the answer tag/block starts to separate reasoning
    reasoning = response.strip()
    if choice_str:
        # First try to find <answer> or <text> tags
        tag_match = re.search(tag_pattern, response, re.DOTALL | re.IGNORECASE)
        if tag_match:
            reasoning = response[: tag_match.start()].strip()
        else:
            # Fall back to manual boxed{} detection with proper brace counting
            boxed_span = _find_boxed_span(response)
            if boxed_span:
                reasoning = response[: boxed_span[0]].strip()

    return reasoning, choice_str


def parse_verifier_verdict(response: str) -> tuple[str, str | None]:
    """
    Parse verifier output (PRM or ORM), extracting reasoning and final verdict.

    Handles multiple verifier output formats:
    - PRM: <step><answer>CORRECT</answer></step> or <step><answer>+1</answer></step>
    - PRM: or <step><answer>Overall Evaluation: ... Therefore the score is: x</answer></step>
    - ORM MCQ/MATH: <answer>CORRECT</answer> or <answer>+1</answer>
    - ORM TEXT: <text>CORRECT</text> or <text>+1</text>

    This parser is modality-agnostic since we only care about CORRECT/INCORRECT.

    Args:
        response: The verifier response text

    Returns:
        A tuple of (reasoning, verdict) where:
        - reasoning: The full verifier output
        - verdict: "CORRECT", "INCORRECT", or None if not found
    """
    if not response:
        return "", None

    normalized_response = response.translate(_UNICODE_SIGN_TABLE)

    def normalize_verdict_token(token: str) -> str | None:
        token_stripped = token.strip().translate(_UNICODE_SIGN_TABLE)
        token_upper = token_stripped.upper()
        if token_upper in {"CORRECT", "INCORRECT"}:
            return token_upper
        # +1 = CORRECT, 0 or -1 = INCORRECT (0 means uncertain/wrong for final answer)
        if token_stripped in {"+1", "1"}:
            return "CORRECT"
        if token_stripped in {"-1", "0"}:
            return "INCORRECT"
        return None

    # Look for PRM-style Likert-scale numeric scores inside <answer> tags
    # Pattern: <answer>... Therefore the score is: x</answer> (x in 1-5)
    score_pattern = re.compile(
        r"<answer>.*?score\s+is:\s*([1-5])\s*</answer>",
        re.IGNORECASE | re.DOTALL,
    )
    match = score_pattern.search(response)
    if match:
        return response.strip(), match.group(1)

    # Fallback: untagged numeric score phrased as "Therefore the score is: X"
    inline_score_match = re.search(r"score\s+is:\s*([1-5])\b", response, re.IGNORECASE)
    if inline_score_match:
        return response.strip(), inline_score_match.group(1)

    # Look for verdict in <answer> or <text> tags
    # Pattern: <answer>CORRECT</answer> or <text>1</text> or <text>+1</text>
    verdict_pattern = re.compile(
        r"<(?:answer|text)>\s*(CORRECT|INCORRECT|\+?1|-1)\s*</(?:answer|text)>",
        re.IGNORECASE,
    )
    match = verdict_pattern.search(normalized_response)

    if match:
        verdict = normalize_verdict_token(match.group(1))
        return response.strip(), verdict

    # PRM sometimes provides a final answer check as a <step> instead of a final <answer>.
    # Handle both "FINAL ANSWER CHECK: +1" and "FINAL ANSWER CHECK +1" formats
    # Also handle <step i="k"> variant and 0 as neutral verdict
    final_check_pattern = re.compile(
        r"<step[^>]*>\s*FINAL ANSWER CHECK:?\s*([+-]?[01])\s*</step>",
        re.IGNORECASE,
    )
    match = final_check_pattern.search(normalized_response)
    if match:
        verdict = normalize_verdict_token(match.group(1))
        return response.strip(), verdict

    # Fallback for untagged final answer check (with optional colon)
    match = re.search(
        r"FINAL ANSWER CHECK:?\s*([+-]?[01])\s*$", normalized_response, re.IGNORECASE
    )
    if match:
        verdict = normalize_verdict_token(match.group(1))
        return response.strip(), verdict

    # Fallback: check for standalone CORRECT/INCORRECT at end of response
    # (in case model doesn't use tags)
    lines = response.strip().split("\n")
    if lines:
        last_line = lines[-1].strip()
        last_line_upper = last_line.upper()
        if re.search(r"\bINCORRECT\b", last_line_upper):
            return response.strip(), "INCORRECT"
        if re.search(r"\bNOT\s+CORRECT\b", last_line_upper):
            return response.strip(), "INCORRECT"
        if re.search(r"\bCORRECT\b", last_line_upper):
            return response.strip(), "CORRECT"
        # Check for numeric verdict at end of line
        token_match = re.search(r"(?:^|\s|\()([+-]?1)\s*$", last_line)
        if token_match:
            verdict = normalize_verdict_token(token_match.group(1))
            return response.strip(), verdict

    # No verdict found
    return response.strip(), None


STEP_BLOCK_WITH_TAIL_RE = re.compile(
    # Allow common Unicode minus/plus variants in the trailing verdict token.
    r"(<step[^>]*>.*?</step>)\s*([\+\-\u2212\u2010\u2011\u2012\u2013\u2014\uFE63\uFF0D\uFF0B]?1|0)?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _extract_step_blocks_with_tail(feedback: str) -> list[tuple[str, str | None]]:
    """
    Extract <step>...</step> blocks with an optional trailing numeric verdict token.

    Some models occasionally emit verdict tokens *after* the closing </step>, e.g.:
        <step i="1">ok +1</step> +1

    To handle this robustly, we scan line-by-line first (the prompt asks for 1 step/line),
    and fall back to a whole-text scan if needed.
    """
    blocks: list[tuple[str, str | None]] = []
    if not feedback:
        return blocks

    for raw_line in feedback.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("<step"):
            continue
        m = STEP_BLOCK_WITH_TAIL_RE.match(line)
        if m:
            blocks.append((m.group(1), m.group(2)))
        else:
            # Fallback: keep the raw line even if it doesn't match the strict pattern.
            blocks.append((line, None))

    if blocks:
        return blocks

    for m in re.finditer(r"<step[^>]*>.*?</step>", feedback, re.IGNORECASE | re.DOTALL):
        blocks.append((m.group(0), None))
    return blocks


def _inner_of_step(step_block: str) -> str:
    m = re.search(r"<step[^>]*>(.*?)</step>", step_block, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else step_block).strip()


def _verdict_from_text(text: str) -> int | None:
    """
    Extract a PRM-style verdict from text.

    Returns:
        1  for correct / +1
        0  for neutral / 0
        -1 for incorrect / -1
        None if unknown
    """
    if text is None:
        return None

    text = text.strip().translate(_UNICODE_SIGN_TABLE)
    # Guardrail: models sometimes copy the format spec literally (e.g. "+1|0|-1")
    # which would otherwise be mis-parsed as a trailing "-1" verdict.
    if re.search(r"\+1\s*\|\s*0\s*\|\s*-1\s*$", text):
        return None

    # Prefer explicit numeric tokens at end.
    m = re.search(r"(?:^|\s)([+-]?1|0)\s*$", text)
    if not m:
        # Fallback: allow tokens glued to text, but avoid arithmetic like "2-1" or "+1|0|-1".
        m = re.search(r"(?<![0-9|])([+-]?1|0)\s*$", text)
    if m:
        tok = m.group(1)
        if tok in {"+1", "1"}:
            return 1
        if tok == "-1":
            return -1
        if tok == "0":
            return 0

    # Fallback: keyword match.
    up = text.upper()
    if re.search(r"\bINCORRECT\b", up):
        return -1
    if re.search(r"\bCORRECT\b", up):
        return 1
    return None


def extract_prm_step_scores(feedback: str) -> list[int | None]:
    """
    Extract per-step scores from PRM verifier feedback.

    The PRM verifier outputs steps in the format:
        <step i="k">note <= 12 words ... +1|0|-1</step>
    
    Also handles older formats:
        <step>...text... | score: X</step>  (1-5 scale)
        <step>...text... +1/-1</step>

    Returns:
        List of integer scores for each step. Uses:
        - +1/0/-1 for new format
        - 1-5 for old format  
        - None if score couldn't be extracted from a step
        
    Note: The FINAL ANSWER CHECK step is excluded from the returned list.
    """
    if not feedback:
        return []

    step_blocks = _extract_step_blocks_with_tail(feedback)
    scores: list[int | None] = []

    for step_block, tail_tok in step_blocks:
        step_stripped = _inner_of_step(step_block)
        # Skip FINAL ANSWER CHECK steps
        if step_stripped.upper().startswith("FINAL ANSWER CHECK"):
            continue

        score: int | None = None

        # Try new format: +1/0/-1 at end of step
        # Pattern: content ending with +1, 0, or -1
        score = _verdict_from_text(tail_tok) if tail_tok is not None else None
        if score is None:
            score = _verdict_from_text(step_stripped)
        
        # Try older format: "| score: X" (1-5 scale)
        if score is None:
            match = re.search(r'\|\s*score:\s*(\d+)', step_stripped, re.IGNORECASE)
            if match:
                score = int(match.group(1))
        
        # Try "Therefore, the score is: X" format
        if score is None:
            match = re.search(r'score\s+is[:\s]+(\d+)', step_stripped, re.IGNORECASE)
            if match:
                score = int(match.group(1))

        scores.append(score)

    return scores


def compute_prm_score(feedback: str) -> float:
    """
    Compute a scalar PRM score in [0, 1] from verifier output.

    Analyzes per-step verdicts (+1/0/-1 or CORRECT/INCORRECT) and returns
    a continuous score. The FINAL ANSWER CHECK verdict is weighted heavily
    to ensure wrong final answers get penalized even if steps look correct.

    Scoring formula:
    - If final verdict is INCORRECT: score = step_ratio * 0.3 (cap at 0.3)
    - If final verdict is CORRECT: score = 0.5 + step_ratio * 0.5 (floor at 0.5)
    - If no final verdict: use step ratio alone

    This ensures wrong answers can't get high scores just from good-looking steps.

    Returns:
        Float in [0, 1] where 1.0 = all steps + final correct, 0.0 = all wrong
    """
    if not feedback:
        return 0.5

    # Count per-step verdicts for nuanced scoring
    step_blocks = _extract_step_blocks_with_tail(feedback)

    correct_count = 0
    incorrect_count = 0
    neutral_count = 0
    final_answer_verdict: int | None = None
    
    for step_block, tail_tok in step_blocks:
        step_stripped = _inner_of_step(step_block)
        # Extract FINAL ANSWER CHECK verdict separately
        if step_stripped.upper().startswith("FINAL ANSWER CHECK"):
            v = _verdict_from_text(tail_tok) if tail_tok is not None else None
            if v is None:
                v = _verdict_from_text(step_stripped)
            if v is not None:
                final_answer_verdict = v
            continue
        verdict = _verdict_from_text(tail_tok) if tail_tok is not None else None
        if verdict is None:
            verdict = _verdict_from_text(step_stripped)
        if verdict is None:
            continue
        if verdict > 0:
            correct_count += 1
        elif verdict < 0:
            incorrect_count += 1
        else:
            neutral_count += 1

    total = correct_count + incorrect_count + neutral_count
    
    # Calculate step ratio (proportion of correct steps)
    if total > 0:
        step_ratio = (correct_count + 0.5 * neutral_count) / total
    else:
        step_ratio = 0.5  # No step info, neutral

    # Get final verdict from <answer> tag if not found in FINAL ANSWER CHECK
    if final_answer_verdict is None:
        _, parsed_verdict = parse_verifier_verdict(feedback)
        if parsed_verdict == "CORRECT":
            final_answer_verdict = 1
        elif parsed_verdict == "INCORRECT":
            final_answer_verdict = -1

    # Combine step ratio with final verdict
    if final_answer_verdict is not None:
        if final_answer_verdict < 0:
            # Wrong final answer: cap score at 0.3 regardless of step quality
            return step_ratio * 0.3
        elif final_answer_verdict > 0:
            # Correct final answer: floor at 0.5, scale up with step quality
            return 0.5 + step_ratio * 0.5
        else:
            # Neutral (0): slight penalty
            return step_ratio * 0.6

    # No final verdict found, use step ratio with slight penalty for uncertainty
    if "VERIFICATION_FAILED" in feedback:
        return 0.1  # Penalize failures
    
    return step_ratio * 0.8  # Slight penalty for missing final verdict
