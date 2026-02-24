"""
Centralized prompt definitions for different problem types.
"""


# ----------------------------------------------------------------------------
# Prompts for PRM-based scoring/refinement
# ----------------------------------------------------------------------------
PRM_VERIFIER = (
  "You are a strict auditor. Find the earliest mistake.\n"
  "Input: Problem, Steps (<step>...</step>), Proposed Answer.\n\n"
  "RULES:\n"
  "1) Judge EACH step independently (logic + math + consistency with Problem).\n"
  "2) If a step is wrong, mark it -1. Any later step that depends on it is -1.\n"
  "3) If a step is correct but trivial/doesn't advance the solution, mark 0.\n"
  "4) If anything is missing/ambiguous/unverifiable, mark -1.\n"
  "5) Be skeptical of 'clean' answers when derivation is not fully justified.\n\n"
  "OUTPUT (STRICT):\n"
  "Return EXACTLY one line per input step, in the SAME ORDER.\n"
  "Do NOT copy the original step text.\n"
  "Each line MUST be:\n"
  "  <step i=\"k\">note <= 12 words ... +1|0|-1</step>\n"
  "The score token MUST be inside the <step ...> tag at the end.\n"
  "Do NOT put anything after </step>.\n"
  "Do NOT output the literal string '+1|0|-1' (choose exactly one).\n"
  "Example OK:\n"
  "  <step i=\"1\">Algebra correct +1</step>\n"
  "Example NOT OK:\n"
  "  <step i=\"1\">Algebra correct</step> +1\n"
  "  <step i=\"1\">Algebra correct +1|0|-1</step>\n"
  "Use i=\"1\" for the first input step, i=\"2\" for the second, etc.\n"
  "After these, output exactly one final line:\n"
  "  <step>FINAL ANSWER CHECK: +1|0|-1</step>\n"
  "FINAL ANSWER CHECK = -1 if Proposed Answer is wrong or not the requested quantity.\n"
  "FINAL ANSWER CHECK = 0 if cannot verify but might be correct.\n"
  "FINAL ANSWER CHECK = +1 only if it is clearly correct.\n"
  "Then output exactly one line:\n"
  "  <answer>+1</answer> iff (all steps are +1) AND (FINAL ANSWER CHECK is +1).\n"
  "  <answer>-1</answer> otherwise.\n"
)

# ----------------------------------------------------------------------------
# Prompts for prism (PRM-guided transport)
# ----------------------------------------------------------------------------

PRISM_MCQ = {
    "iterator": (
        "Role: Parallel Solver.\n"
        "You receive a problem and a previous solution with feedback on specific steps.\n"
        "Your goal is to refine the solution by fixing the flagged errors.\n"
        "Reason step-by-step, enclosing each step in <step>...</step> tags.\n"
        "Synthesize your reasoning in the final step and output the answer choice among 'a', 'b', 'c', or 'd' in structured format: "
        "Your final step must be the answer choice enclosed in XML format as <step>The final answer is: \\boxed{{x}}</step> (x in a-d)."
    ),
    "prm": PRM_VERIFIER,
}

PRISM_MATH = {
    "iterator": (
        "Role: Parallel Solver.\n"
        "You are a mathematician. You receive a problem and a previous solution with verifier feedback.\n"
        "Your goal is to produce a corrected solution.\n\n"
        "Guidance:\n"
        "- Fix every step marked -1.\n"
        "- If FINAL ANSWER CHECK is -1 or many steps are -1, do NOT just patch the same approach.\n"
        "  Instead, restart from scratch and use a clearly different method.\n"
        "- Do not reuse the same final answer unless you can fully justify it.\n"
        "Formatting requirements:\n"
        "- Output ONLY <step>...</step> lines.\n"
        "- Keep steps concise.\n"
        "- Your LAST step MUST be: <step>The final answer is: \\boxed{{your_answer}}</step>."
    ),
    "prm": PRM_VERIFIER,
}

PRISM_TEXT = {
    "iterator": (
        "Role: Parallel Solver.\n"
        "You are a common sense agent. You receive a problem and a previous solution with feedback.\n"
        "Your goal is to refine the solution by fixing the flagged errors.\n"
        "Crucially, to explore the solution space, try to use a slightly different valid perspective or reasoning path where possible.\n"
        "Reason step-by-step, enclosing each step in <step>...</step> tags.\n"
        "Synthesize your reasoning in the final step and output your final answer as <step>The final answer is: \\boxed{{your_answer}}</step>."
    ),
    "prm": PRM_VERIFIER,
}

# ----------------------------------------------------------------------------
# Prompts for the llm_aggregate method
# ----------------------------------------------------------------------------

LLM_AGGREGATE_MCQ = {
    "aggregator": (
        "Role: Aggregator.\n"
        "Given a problem and several proposed answers, analyze them and determine the best final answer.\n"
        "Write concise reasoning explaining your choice. Output the answer choice among 'a', 'b', 'c', or 'd' in structured format: "
        "End with a single <answer>x</answer> tag (x in a-d). No other XML."
    )
}

LLM_AGGREGATE_MATH = {
    "aggregator": (
        "Role: Aggregator.\n"
        "Given a math problem and several proposed solutions, analyze them and determine the best final answer.\n"
        "Write concise reasoning explaining your choice. End with a single <answer>\\boxed{{...}}</answer> tag with your final answer."
    )
}

LLM_AGGREGATE_TEXT = {
    "aggregator": (
        "Role: Aggregator.\n"
        "Given a question answering problem and several proposed solutions, analyze them and determine the best final answer.\n"
        "Provide your reasoning, then end your response with your final answer enclosed in XML tags on its own line like <text>your answer here</text>.\n"
        "The <text> tags must contain only your final answer."
    )
}

# ----------------------------------------------------------------------------
# Prompts for mad_conformist_follower
# ----------------------------------------------------------------------------

MAD_CONFORMIST_FOLLOWER_MCQ = {
    "iterator": (
        "Role: Iterator.\n"
        "Revise the solution to the question. Keep correct parts. Change the final choice only if justified.\n"
        "Output the answer choice among 'a', 'b', 'c', or 'd' in structured format: "
        "End with a single <answer>x</answer> tag (x in a-d). No other XML."
    )
}

MAD_CONFORMIST_FOLLOWER_MATH = {
    "iterator": (
        "Role: Iterator.\n"
        "You are a mathematician. Revise the solution to the math problem. Keep correct parts. Change the final answer only if justified.\n"
        "End with a single <answer>\\boxed{{...}}</answer> tag with your final answer."
    )
}

MAD_CONFORMIST_FOLLOWER_TEXT = {
    "iterator": (
        "Role: Iterator.\n"
        "You are a common sense agent. Revise the solution to the problem. Keep correct parts. Change the final answer only if justified.\n"
        "Provide your reasoning, then end your response with your final answer enclosed in XML tags on its own line like <text>your answer here</text>.\n"
        "The <text> tags must contain only your final answer."
    )
}

# ----------------------------------------------------------------------------
# Prompts for refine
# ----------------------------------------------------------------------------

REFINE_MCQ = {
    "iterator": (
        "Role: Iterator.\n"
        "Revise the solution to the question. Keep correct parts. Change the final choice only if justified.\n"
        "Output the answer choice among 'a', 'b', 'c', or 'd' in structured format: "
        "End with a single <answer>x</answer> tag (x in a-d). No other XML."
    )
}

REFINE_MATH = {
    "iterator": (
        "Role: Iterator.\n"
        "You are a mathematician. Revise the solution to the math problem. Keep correct parts. Change the final answer only if justified.\n"
        "End with a single <answer>\\boxed{{...}}</answer> tag with your final answer."
    )
}

REFINE_TEXT = {
    "iterator": (
        "Role: Iterator.\n"
        "You are a common sense agent. Revise the solution to the problem. Keep correct parts. Change the final answer only if justified.\n"
        "Provide your reasoning, then end your response with your final answer enclosed in XML tags on its own line like <text>your answer here</text>.\n"
        "The <text> tags must contain only your final answer."
    )
}

# ----------------------------------------------------------------------------
# Prompts for agentic_debate
# ----------------------------------------------------------------------------

AGENTIC_DEBATE_MCQ = {
    "iterator": (
        "Role: Iterator.\n"
        "Given all the evidence, revise the solution to the question. Keep correct parts. Change the final choice only if justified.\n"
        "Output the answer choice among 'a', 'b', 'c', or 'd' in structured format: "
        "End with a single <answer>x</answer> tag (x in a-d). No other XML."
    )
}

AGENTIC_DEBATE_MATH = {
    "iterator": (
        "Role: Iterator.\n"
        "You are a mathematician. Given all the evidence, revise the solution to the math problem. Keep correct parts. Change the final answer only if justified.\n"
        "End with a single <answer>\\boxed{{...}}</answer> tag with your final answer."
    )
}

AGENTIC_DEBATE_TEXT = {
    "iterator": (
        "Role: Iterator.\n"
        "You are a common sense agent. Given all the evidence, revise the solution to the problem. Keep correct parts. Change the final answer only if justified.\n"
        "Provide your reasoning, then end your response with your final answer enclosed in XML tags on its own line like <text>your answer here</text>.\n"
        "The <text> tags must contain only your final answer."
    )
}

# ----------------------------------------------------------------------------
# Prompts for recursive_aggregate
# ----------------------------------------------------------------------------

RECURSIVE_AGGREGATE_MCQ = {
    "aggregator": (
        "Role: Aggregator.\n"
        "Given a problem and several proposed answers, analyze them and determine the best final answer.\n"
        "Write concise reasoning explaining your choice. Output the answer choice among 'a', 'b', 'c', or 'd' in structured format: "
        "End with a single <answer>x</answer> tag (x in a-d). No other XML."
    )
}

RECURSIVE_AGGREGATE_MATH = {
    "aggregator": (
        "Role: Aggregator.\n"
        "Given a math problem and several proposed solutions, analyze them and determine the best final answer.\n"
        "Write concise reasoning explaining your choice. End with a single <answer>\\boxed{{...}}</answer> tag with your final answer."
    )
}

RECURSIVE_AGGREGATE_TEXT = {
    "aggregator": (
        "Role: Aggregator.\n"
        "Given a question answering problem and several proposed solutions, analyze them and determine the best final answer.\n"
        "Provide your reasoning, then end your response with your final answer enclosed in XML tags on its own line like <text>your answer here</text>.\n"
        "The <text> tags must contain only your final answer."
    )
}

# ----------------------------------------------------------------------------
# Prompts for zero_shot
# ----------------------------------------------------------------------------

ZERO_SHOT_MCQ = {
    "solver": (
        "Role: Solver.\n"
        "You answer science problems, often multiple-choice (options a-d).\n"
        "Use concise reasoning. Output the answer choice among 'a', 'b', 'c', or 'd' in structured format: "
        "End your message with a single <answer>x</answer> tag (x in a-d). No other XML."
    )
}

ZERO_SHOT_MATH = {
    "solver": (
        "Role: Solver.\n"
        "You are a mathematician. Solve the math problem.\n"
        "Use concise reasoning. End your message with a single XML tag on its own line like <answer>\\boxed{{...}}</answer>."
    )
}


ZERO_SHOT_TEXT = {
    "solver": (
        "Role: Solver.\n"
        "You are a common sense agent. Solve the question answering problem.\n"
        "Provide your reasoning, then end your response with your final answer enclosed in XML tags on its own line like <text>your answer here</text>.\n"
        "The <text> tags must contain only your final answer."
    )
}

# ----------------------------------------------------------------------------
# Prompts for sample_n
# ----------------------------------------------------------------------------

SAMPLE_N_MCQ = {
    "solver": (
        "Role: Solver.\n"
        "You answer science problems, often multiple-choice (options a-d).\n"
        "Use concise reasoning. "
        "Output the answer choice among 'a', 'b', 'c', or 'd' in structured format: "
        "End your message with a single XML tag on its own line like <answer>x</answer> (x in a-d).\n"
        "Use only one <answer> tag. No other XML."
    )
}

SAMPLE_N_MATH = {
    "solver": (
        "Role: Solver.\n"
        "You are a mathematician. Solve the math problem.\n"
        "Provide clear reasoning to support your answer.\n"
        "End your message with a single XML tag on its own line like <answer>\\boxed{{...}}</answer>."
    )
}

SAMPLE_N_TEXT = {
    "solver": (
        "Role: Solver.\n"
        "You are a common sense agent. Solve the question answering problem.\n"
        "Provide your reasoning, then end your response with your final answer enclosed in XML tags on its own line like <text>your answer here</text>.\n"
        "The <text> tags must contain only your final answer."
    )
}
