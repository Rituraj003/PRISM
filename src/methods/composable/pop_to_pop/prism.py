from __future__ import annotations
import asyncio
import math
import random
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence, cast
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from tqdm.asyncio import tqdm as atqdm
from answer_verification import (
    compute_prm_score,
    normalize_answer,
    parse_answer_with_reasoning,
    parse_verifier_verdict,
)
from prompts import (
    PRISM_MATH,
    PRISM_MCQ,
    PRISM_TEXT,
)
from shared import QuestionType, build_model
from ..stepwise import coerce_answer_stepwise
from ..stages import Answer, PopulationToPopulation, StageContext
# Prompt for head-to-head comparison when two answers both claim 1.0 score
COMPARE_SOLUTIONS_PROMPT = (
    "You are a careful judge. Two solutions to the same problem both claim to be correct.\n"
    "Compare them and determine which one (if any) is actually correct.\n\n"
    "Problem:\n{problem}\n\n"
    "Solution A (answer: {choice_a}):\n{reasoning_a}\n\n"
    "Solution B (answer: {choice_b}):\n{reasoning_b}\n\n"
    "Instructions:\n"
    "1. Check each solution's reasoning and final answer.\n"
    "2. If Solution A is correct and B is wrong, output: <verdict>A</verdict>\n"
    "3. If Solution B is correct and A is wrong, output: <verdict>B</verdict>\n"
    "4. If BOTH are wrong or you cannot determine, output: <verdict>NEITHER</verdict>\n\n"
    "Output exactly one <verdict>...</verdict> tag with your decision."
)


def parse_answer(response: str, question_type: QuestionType) -> Answer:
    """
    Parse answer from response for PRM-style stepwise methods.
    IMPORTANT: Keep the full raw response as `reasoning` so we preserve well-formed
    <step>...</step> blocks (including the final step containing \\boxed{...}).
    `parse_answer_with_reasoning()` intentionally truncates reasoning before \\boxed{...},
    which would break the step structure and confuse the verifier.
    """
    response_text = response.strip() if response else ""
    _, choice_str = parse_answer_with_reasoning(response_text, question_type=question_type)
    return Answer(response_text, choice_str)
@dataclass
class ScoredParticle:
    """A particle with its PRM score and verification feedback."""
    answer: Answer
    score: float # Scalar PRM score in [0, 1]
    feedback: str # Textual feedback for refinement
    weight: float = 1.0 # Unnormalized importance weight
@dataclass(frozen=True)
class PrismConfig:
    temperature: float = 0.7
    ess_threshold: float = 0.5
    acceptance_noise: float = 0.1
    arbitration_score_clamp: float = 0.3
    all_bad_score_threshold: float = 1e-6
    max_attempts: int = 2
    perfect_score_epsilon: float = 1e-6
    verification_fail_score: float = 1e-6
    verifier_timeout_s: float = 300.0
    proposal_timeout_s: float = 300.0
    max_copies_fraction: float | None = 0.3
    debug: bool = False
@dataclass(frozen=True)
class ArbitrationDecision:
    reason: str
    choice_a: str
    choice_b: str
def score_to_weight(score: float, temperature: float) -> float:
    """
    Compute importance weight from PRM score.
    w = score^(1/T), with exact 0 for non-positive scores.
    """
    if score <= 0:
        return 0.0
    return math.exp(math.log(score) / max(temperature, 1e-6))
def acceptance_decision(
    current_weight: float,
    proposal_weight: float,
    proposal_score: float,
    current_score: float,
    rng: random.Random | None = None,
) -> tuple[bool, float | None]:
    if current_weight > 0:
        acceptance_ratio = proposal_weight / current_weight
        threshold = min(1.0, acceptance_ratio)
        draw = (rng or random).random()
        return draw < threshold, acceptance_ratio
    return proposal_score > current_score, None
def extract_steps(reasoning: str) -> list[str]:
    """Extract reasoning steps from various formats."""
    if not reasoning or not reasoning.strip():
        return []
    step_blocks = re.findall(
        r"<step[^>]*>(.*?)</step>", reasoning, re.DOTALL | re.IGNORECASE
    )
    if step_blocks:
        steps = [step.strip() for step in step_blocks if step.strip()]
        if steps:
            return steps
    cleaned = re.sub(r"</?answer[^>]*>", "", reasoning, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?text[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return []
    numbered_pattern = (
        r"(?:^|\n)\s*(?:(?:Step\s*)?\d+[\.\):]|\([a-z]\)|\([0-9]+\))\s*"
        r"(.+?)(?=(?:\n\s*(?:(?:Step\s*)?\d+[\.\):]|\([a-z]\)|\([0-9]+\)))|$)"
    )
    numbered_matches = re.findall(numbered_pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if len(numbered_matches) >= 2:
        steps = [m.strip() for m in numbered_matches if m.strip()]
        if len(steps) >= 2:
            return steps
    bullet_pattern = r"(?:^|\n)\s*[-*•]\s*(.+?)(?=(?:\n\s*[-*•])|$)"
    bullet_matches = re.findall(bullet_pattern, cleaned, re.DOTALL)
    if len(bullet_matches) >= 2:
        steps = [m.strip() for m in bullet_matches if m.strip()]
        if len(steps) >= 2:
            return steps
    paragraphs = [s.strip() for s in cleaned.split("\n\n") if s.strip()]
    if len(paragraphs) >= 2:
        return paragraphs
    lines = [s.strip() for s in cleaned.split("\n") if s.strip() and len(s.strip()) > 10]
    if len(lines) >= 2:
        return lines
    if len(cleaned) > 100:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        if len(sentences) >= 2:
            return sentences
    if len(cleaned) > 20:
        return [cleaned]
    return []
def make_cache_key(answer: Answer) -> str:
    """Create a deterministic cache key from answer reasoning and choice."""
    reasoning = (answer.reasoning or "").encode("utf-8")
    choice = (answer.choice or "").encode("utf-8")
    return f"{zlib.crc32(reasoning)}:{zlib.crc32(choice)}"
def choice_key(answer: Answer, question_type: QuestionType) -> str:
    normalized = normalize_answer(answer.choice, question_type)
    if normalized is not None:
        return normalized
    return str(answer.choice) if answer.choice is not None else "<none>"


def particle_key(particle: ScoredParticle, question_type: QuestionType) -> str:
    return choice_key(particle.answer, question_type)
def group_particles_by_choice(
    particles: Sequence[ScoredParticle],
    question_type: QuestionType,
) -> dict[str, list[ScoredParticle]]:
    by_choice: dict[str, list[ScoredParticle]] = {}
    for particle in particles:
        key = particle_key(particle, question_type)
        by_choice.setdefault(key, []).append(particle)
    return by_choice
def select_arbitration_candidates(
    by_choice_all: dict[str, list[ScoredParticle]],
    by_choice_perfect: dict[str, list[ScoredParticle]],
    config: PrismConfig,
) -> ArbitrationDecision | None:
    if len(by_choice_all) < 2:
        return None
    conflicting_perfect = len(by_choice_perfect) >= 2
    if conflicting_perfect:
        sorted_choices = sorted(
            by_choice_perfect.keys(),
            key=lambda k: len(by_choice_perfect[k]),
            reverse=True,
        )
        return ArbitrationDecision(
            reason="conflicting_perfect_scores",
            choice_a=sorted_choices[0],
            choice_b=sorted_choices[1],
        )
    return None
def systematic_resample(particles: list[ScoredParticle], n_samples: int) -> list[int]:
    """
    Systematic resampling for SMC/particle filtering.
    Returns indices of selected particles.
    Uses a single random offset U ~ Uniform(0, 1/n_samples) for all positions,
    which is the defining property of systematic resampling.
    """
    weights = [p.weight for p in particles]
    total_weight = sum(weights)
    if total_weight == 0:
        return [i % len(particles) for i in range(n_samples)]
    normalized = [w / total_weight for w in weights]
    cdf = []
    cumsum = 0.0
    for w in normalized:
        cumsum += w
        cdf.append(cumsum)
    u = random.random() / n_samples
    positions = [u + i / n_samples for i in range(n_samples)]
    indices = []
    j = 0
    for pos in positions:
        while j < len(cdf) - 1 and cdf[j] < pos:
            j += 1
        indices.append(j)
    return indices
def compute_ess(weights: list[float]) -> float:
    """
    Compute Effective Sample Size (ESS).
    ESS = (Σw)² / Σ(w²)
    Low ESS indicates weight degeneracy (need resampling).
    """
    total = sum(weights)
    if total == 0:
        return 0.0
    sum_sq = sum(w * w for w in weights)
    if sum_sq == 0:
        return len(weights)
    return (total * total) / sum_sq
@dataclass(frozen=True)
class _PrismState:
    context: StageContext
    question: str
    question_type: QuestionType
    verifier_agent: Agent[None, str]
    iterator_agent: Agent[None, str]
    compare_agent: Agent[None, str]
    verifier_settings: ModelSettings
    use_secondary_verifier_usage: bool
    score_cache: dict[str, tuple[float, str]]
    depth_iter: int
    temperature: float
class Prism(PopulationToPopulation):
    """
    PRM-guided Sequential Monte Carlo (SMC) with heuristic refinement.
    Implements SMC mechanics with LLM-guided proposals:
    - Scalar PRM scores as energy function: E(τ) = -log(PRM(τ))
    - Boltzmann weights: w(τ) ∝ score(τ)^(1/T)
    - Systematic resampling when ESS drops (population-level interaction)
    - Heuristic acceptance: accept proposals stochastically based on relative weights
      (Note: This is NOT true Metropolis-Hastings because the LLM proposal
       kernel q(τ'|τ) is not symmetric and is intractable to estimate.
       We use a soft weight-ratio acceptance to allow worse proposals occasionally.)
    - Temperature-controlled exploration/exploitation
    Target distribution: π(τ) ∝ score(τ)^(1/T)
    """
    def __init__(
        self,
        temperature: float = 0.7,
        ess_threshold: float = 0.5,
        acceptance_noise: float = 0.1,
        config: PrismConfig | None = None,
    ) -> None:
        if config is None:
            config = PrismConfig(
                temperature=temperature,
                ess_threshold=ess_threshold,
                acceptance_noise=acceptance_noise,
            )
        self.cfg = config
    async def __call__(
        self, context: StageContext, population: list[Answer]
    ) -> list[Answer]:
        question_type = context.question_type
        prompts = self._select_prompts(question_type)
        if context.settings.verifier_model_name:
            verifier_model, verifier_model_settings_base = build_model(
                context.settings, verifier=True
            )
            use_secondary_verifier_usage = True
        else:
            verifier_model = context.model
            verifier_model_settings_base = context.model_settings
            use_secondary_verifier_usage = False

        verifier_settings = self._build_verifier_settings(verifier_model_settings_base)
        verifier_agent: Agent[None, str] = Agent(
            verifier_model,
            system_prompt=prompts["prm"],
            model_settings=verifier_settings,
        )
        iterator_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["iterator"],
            model_settings=context.model_settings,
        )
        compare_agent: Agent[None, str] = Agent(
            verifier_model,
            system_prompt=(
                "Output ONLY: <verdict>A</verdict> or <verdict>B</verdict> or "
                "<verdict>NEITHER</verdict>."
            ),
            model_settings=verifier_settings,
        )
        question = context.example.to_prompt()
        n_particles = len(population)
        depth_iter = self._next_depth_iter(context)
        temperature = self._temperature_for_depth(depth_iter)
        score_cache = self._get_score_cache(context)
        state = _PrismState(
            context=context,
            question=question,
            question_type=question_type,
            verifier_agent=verifier_agent,
            iterator_agent=iterator_agent,
            compare_agent=compare_agent,
            verifier_settings=verifier_settings,
            use_secondary_verifier_usage=use_secondary_verifier_usage,
            score_cache=score_cache,
            depth_iter=depth_iter,
            temperature=temperature,
        )
        # Enforce canonical <step> format early; extract_steps is fallback-only.
        population = [coerce_answer_stepwise(answer) for answer in population]
        particles = await self.score_population(state, population)
        particles = await self.arbitrate_conflicts(state, particles)
        self.compute_weights(particles, state.temperature)
        particles = self.maybe_resample(state, particles, n_particles)
        self.compute_weights(particles, state.temperature)
        particles = await self.mutate_population(state, particles)
        self.compute_weights(particles, state.temperature)
        new_population = await self.rejuvenate(state, particles)
        return list(new_population)
    @staticmethod
    def _select_prompts(question_type: QuestionType) -> dict[str, str]:
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            return PRISM_MCQ
        if prompt_family == "math":
            return PRISM_MATH
        return PRISM_TEXT
    @staticmethod
    def _build_verifier_settings(model_settings: ModelSettings) -> ModelSettings:
        verifier_settings_dict = dict(model_settings)
        verifier_settings_dict["temperature"] = 0.0
        verifier_settings_dict["top_p"] = 1.0
        return cast(ModelSettings, verifier_settings_dict)
    @staticmethod
    def _record_verifier_usage(state: _PrismState, usage: RunUsage) -> None:
        if state.use_secondary_verifier_usage:
            state.context.record_secondary_usage(usage)
            return
        state.context.record_usage(usage)
    @staticmethod
    def _next_depth_iter(context: StageContext) -> int:
        iter_key = "prism_depth_iter"
        try:
            depth_iter = int(context.cache.get(iter_key, 0) or 0)
        except Exception:
            depth_iter = 0
        context.cache[iter_key] = depth_iter + 1
        return depth_iter
    def _temperature_for_depth(self, depth_iter: int) -> float:
        _ = depth_iter
        return self.cfg.temperature
    @staticmethod
    def _get_score_cache(context: StageContext) -> dict[str, tuple[float, str]]:
        prism_cache_key = "prism_score_cache"
        if prism_cache_key not in context.cache:
            context.cache[prism_cache_key] = {}
        return context.cache[prism_cache_key]
    @staticmethod
    def _verifier_fallback_output() -> str:
        return (
            "<step>No steps provided -1</step>\n"
            "<step>FINAL ANSWER CHECK -1</step>\n"
            "<answer>-1</answer>"
        )
    @staticmethod
    def _format_reasoning_for_verifier(reasoning: str) -> str | None:
        if not reasoning or not reasoning.strip():
            return None
        step_blocks = re.findall(
            r"<step[^>]*>(.*?)</step>", reasoning, re.DOTALL | re.IGNORECASE
        )
        if step_blocks:
            steps = [step.strip() for step in step_blocks if step.strip()]
            if steps:
                return "\n".join(f"<step>{s}</step>" for s in steps)
        steps = extract_steps(reasoning)
        if not steps:
            return None
        return "\n".join(f"<step>{s}</step>" for s in steps)
    async def _run_verifier(
        self,
        state: _PrismState,
        reasoning: str,
        proposed_answer: str | None,
    ) -> tuple[str, float]:
        formatted_reasoning = self._format_reasoning_for_verifier(reasoning)
        if not formatted_reasoning:
            feedback = self._verifier_fallback_output()
            return feedback, compute_prm_score(feedback)
        proposed_answer_text = proposed_answer if proposed_answer is not None else "<none>"
        verify_prompt = (
            f"Problem:\n{state.question}\n\n"
            f"Reasoning:\n{formatted_reasoning}\n\n"
            f"Proposed Answer:\n{proposed_answer_text}\n\n"
            "Follow the system instructions exactly. Output only <step> lines and the final "
            "<answer> line."
        )
        try:
            verify_result = await asyncio.wait_for(
                state.verifier_agent.run(verify_prompt),
                timeout=self.cfg.verifier_timeout_s,
            )
            self._record_verifier_usage(state, verify_result.usage())
            feedback = verify_result.output or ""
            if not feedback:
                feedback = self._verifier_fallback_output()
            step_count = len(
                re.findall(
                    r"<step[^>]*>(.*?)</step>", feedback, re.DOTALL | re.IGNORECASE
                )
            )
            _, verdict = parse_verifier_verdict(feedback)
            has_final_check = "FINAL ANSWER CHECK" in feedback.upper()
            if step_count < 2 or verdict is None or not has_final_check:
                feedback = self._verifier_fallback_output()
            return feedback, compute_prm_score(feedback)
        except (asyncio.TimeoutError, TimeoutError):
            pass
        except Exception:
            if self.cfg.debug:
                raise
        feedback = self._verifier_fallback_output()
        return feedback, compute_prm_score(feedback)
    async def score_population(
        self,
        state: _PrismState,
        population: Sequence[Answer],
    ) -> list[ScoredParticle]:
        async def score_particle(answer: Answer, idx: int) -> ScoredParticle:
            if (
                answer.choice is None
                or answer.choice.strip() == ""
                or answer.choice.strip().lower() == "<none>"
            ):
                return ScoredParticle(
                    answer=answer,
                    score=0.0,
                    feedback="<MISSING_ANSWER/>",
                    weight=0.0,
                )
            cache_key = make_cache_key(answer)
            if cache_key in state.score_cache:
                cached_score, cached_feedback = state.score_cache[cache_key]
                weight = score_to_weight(cached_score, state.temperature)
                return ScoredParticle(
                    answer=answer,
                    score=cached_score,
                    feedback=cached_feedback,
                    weight=weight,
                )
            try:
                feedback, score = await self._run_verifier(
                    state,
                    answer.reasoning,
                    answer.choice,
                )
                weight = score_to_weight(score, state.temperature)
                state.score_cache[cache_key] = (score, feedback)
                return ScoredParticle(
                    answer=answer,
                    score=score,
                    feedback=feedback,
                    weight=weight,
                )
            except Exception as exc:
                if self.cfg.debug:
                    raise
                score = self.cfg.verification_fail_score
                return ScoredParticle(
                    answer=answer,
                    score=score,
                    feedback=f"<VERIFICATION_FAILED>{exc}</VERIFICATION_FAILED>",
                    weight=score_to_weight(score, state.temperature),
                )
        score_tasks = [
            score_particle(answer, idx) for idx, answer in enumerate(population)
        ]
        particles = await atqdm.gather(
            *score_tasks,
            desc="prism-score",
            total=len(score_tasks),
            dynamic_ncols=True,
        )
        return list(particles)
    async def arbitrate_conflicts(
        self,
        state: _PrismState,
        particles: list[ScoredParticle],
    ) -> list[ScoredParticle]:
        by_choice_all = group_particles_by_choice(particles, state.question_type)
        perfect_particles = [
            p
            for p in particles
            if p.score >= 1.0 - self.cfg.perfect_score_epsilon
        ]
        by_choice_perfect = group_particles_by_choice(
            perfect_particles,
            state.question_type,
        )
        decision = select_arbitration_candidates(
            by_choice_all,
            by_choice_perfect,
            self.cfg,
        )
        if not decision:
            return particles
        def total_weight(choice: str) -> float:
            return sum(p.weight for p in by_choice_all[choice])
        def best_score(choice: str) -> float:
            return max(p.score for p in by_choice_all[choice])
        weight_ranked = sorted(by_choice_all.keys(), key=total_weight, reverse=True)
        top_w1, top_w2 = weight_ranked[0], weight_ranked[1]
        w1 = total_weight(top_w1)
        w2 = total_weight(top_w2)
        w_ratio = (w2 / w1) if w1 > 0 else 0.0
        score_ranked = sorted(by_choice_all.keys(), key=best_score, reverse=True)
        top_s1, top_s2 = score_ranked[0], score_ranked[1]
        score_gap = best_score(top_s1) - best_score(top_s2)
        choice_a = decision.choice_a
        choice_b = decision.choice_b
        rep_a = max(by_choice_all[choice_a], key=lambda p: (p.score, p.weight))
        rep_b = max(by_choice_all[choice_b], key=lambda p: (p.score, p.weight))
        weight_a = total_weight(choice_a)
        weight_b = total_weight(choice_b)
        weight_ratio = (weight_b / weight_a) if weight_a > 0 else 0.0
        if self.cfg.debug:
            atqdm.write(
                "[prism][arbitration] "
                f"choices={choice_a},{choice_b} "
                f"w_ratio={w_ratio:.4f} score_gap={score_gap:.4f} "
                f"weight_ratio_b_over_a={weight_ratio:.4f}"
            )
        compare_prompt = COMPARE_SOLUTIONS_PROMPT.format(
            problem=state.question,
            choice_a=choice_a,
            reasoning_a=rep_a.answer.reasoning or "",
            choice_b=choice_b,
            reasoning_b=rep_b.answer.reasoning or "",
        )
        try:
            compare_result = await asyncio.wait_for(
                state.compare_agent.run(
                    compare_prompt,
                    model_settings=state.verifier_settings,
                ),
                timeout=self.cfg.verifier_timeout_s,
            )
            self._record_verifier_usage(state, compare_result.usage())
            compare_output = compare_result.output or ""
            verdict_match = re.search(
                r"<verdict>\s*(A|B|NEITHER)\s*</verdict>",
                compare_output,
                re.IGNORECASE,
            )
            if verdict_match:
                verdict = verdict_match.group(1).upper()
                loser_choice: str | None = None
                if verdict == "A":
                    loser_choice = choice_b
                elif verdict == "B":
                    loser_choice = choice_a
                elif verdict == "NEITHER":
                    clamped_particles = by_choice_all[choice_a] + by_choice_all[choice_b]
                    for p in clamped_particles:
                        p.score = min(p.score, self.cfg.arbitration_score_clamp)
                        state.score_cache[make_cache_key(p.answer)] = (
                            p.score,
                            "<ARBITRATION_PENALIZED/>",
                        )
                if loser_choice:
                    clamped_particles = by_choice_all[loser_choice]
                    for p in clamped_particles:
                        p.score = min(p.score, self.cfg.arbitration_score_clamp)
                        state.score_cache[make_cache_key(p.answer)] = (
                            p.score,
                            "<ARBITRATION_PENALIZED/>",
                        )
        except Exception:
            if self.cfg.debug:
                raise
        return particles
    def compute_weights(
        self,
        particles: Iterable[ScoredParticle],
        temperature: float,
    ) -> None:
        for particle in particles:
            particle.weight = score_to_weight(particle.score, temperature)
    def maybe_resample(
        self,
        state: _PrismState,
        particles: list[ScoredParticle],
        n_particles: int,
    ) -> list[ScoredParticle]:
        weights = [p.weight for p in particles]
        ess = compute_ess(weights)
        max_weight = max(weights) if weights else 0.0
        max_score = max((p.score for p in particles), default=0.0)
        all_bad = max_score < self.cfg.all_bad_score_threshold
        log_lines = [
            (
                f"[prism] {state.context.example.dataset}/"
                f"{state.context.example.question_id} iter={state.depth_iter} "
                f"temp={state.temperature:.2f} ess={ess:.2f} "
                f"threshold={self.cfg.ess_threshold:.2f} "
                f"max_w={max_weight:.6f} max_score={max_score:.6f}"
            )
        ]
        scored = sorted(
            [
                (
                    particle_key(p, state.question_type),
                    p.score,
                    p.weight,
                )
                for p in particles
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        for rank, (choice, score, weight) in enumerate(scored, start=1):
            log_lines.append(
                f" #{rank:02d} choice={choice} score={score:.3f} weight={weight:.3f}"
            )
        should_resample = (ess < self.cfg.ess_threshold * n_particles) or all_bad
        if should_resample and all_bad:
            log_lines.append(
                " [all-bad detected: skipping resample, relying on MH exploration]"
            )
            should_resample = False
        if should_resample:
            resample_indices = systematic_resample(particles, n_particles)
            new_particles: list[ScoredParticle] = []
            max_fraction = getattr(
                state.context.settings,
                "follower_ratio",
                self.cfg.max_copies_fraction,
            )
            max_copies = None
            if max_fraction is not None and max_fraction > 0:
                max_fraction = min(max_fraction, 1.0)
                max_copies = max(1, int(math.ceil(n_particles * max_fraction)))
            particle_keys: list[str] = []
            seen_reasonings: Counter[str] = Counter()
            if max_copies is not None:
                def reasoning_key(answer: Answer) -> str:
                    reasoning = (answer.reasoning or "").encode("utf-8")
                    return str(zlib.crc32(reasoning))
                particle_keys = [reasoning_key(p.answer) for p in particles]
            for parent_idx in resample_indices:
                parent = particles[parent_idx]
                if max_copies is not None:
                    reasoning_k = particle_keys[parent_idx]
                    if seen_reasonings[reasoning_k] >= max_copies:
                        fallback_indices = [
                            i
                            for i, key in enumerate(particle_keys)
                            if seen_reasonings[key] < max_copies
                        ]
                        if fallback_indices:
                            parent_idx = random.choice(fallback_indices)
                            parent = particles[parent_idx]
                            reasoning_k = particle_keys[parent_idx]
                    seen_reasonings[reasoning_k] += 1
                new_particles.append(
                    ScoredParticle(
                        answer=parent.answer,
                        score=parent.score,
                        feedback=parent.feedback,
                        weight=1.0,
                    )
                )
            particles = new_particles
            resampled_counts = Counter(
                particle_key(p, state.question_type) for p in particles
            )
            resampled_summary = ", ".join(
                f"{choice}:{count}" for choice, count in resampled_counts.most_common()
            )
            log_lines.append(f" resampled=yes counts={resampled_summary}")
        else:
            log_lines.append(" resampled=no")
        atqdm.write("\n".join(log_lines))
        return particles
    async def rejuvenate(
        self,
        state: _PrismState,
        particles: list[ScoredParticle],
    ) -> list[Answer]:
        mh_tasks = [self._propose_and_accept(state, p, i) for i, p in enumerate(particles)]
        mh_results = await atqdm.gather(
            *mh_tasks,
            desc="prism-mh",
            total=len(mh_tasks),
            dynamic_ncols=True,
        )
        return list(mh_results)
    async def mutate_population(
        self,
        state: _PrismState,
        particles: list[ScoredParticle],
    ) -> list[ScoredParticle]:
        return particles
    async def _generate_fresh_particle(
        self,
        state: _PrismState,
        idx: int,
    ) -> ScoredParticle | None:
        prompt = (
            f"Problem:\n{state.question}\n\n"
            "Solve from scratch. Use a DIFFERENT approach than a standard solution if possible.\n\n"
            "Formatting requirements:\n"
            "- Output ONLY <step>...</step> lines.\n"
            "- Keep steps concise.\n"
            "- LAST step MUST be: <step>The final answer is: \\boxed{...}</step>\n"
        )
        try:
            gen = await asyncio.wait_for(
                state.iterator_agent.run(prompt),
                timeout=self.cfg.proposal_timeout_s,
            )
            state.context.record_usage(gen.usage())
            raw = gen.output or ""
            proposal = parse_answer(raw, question_type=state.question_type)
            proposal = coerce_answer_stepwise(proposal)
            feedback, score = await self._run_verifier(
                state, proposal.reasoning, proposal.choice
            )
            weight = score_to_weight(score, state.temperature)
            key = make_cache_key(proposal)
            state.score_cache[key] = (score, feedback)
            return ScoredParticle(
                answer=proposal, score=score, feedback=feedback, weight=weight
            )
        except Exception:
            if self.cfg.debug:
                raise
            return None
    async def _propose_and_accept(
        self,
        state: _PrismState,
        particle: ScoredParticle,
        idx: int,
    ) -> Answer:
        if particle.score >= 1.0 - self.cfg.perfect_score_epsilon:
            return particle.answer
        try:
            feedback_lines = [
                line
                for line in particle.feedback.split("\n")
                if "INCORRECT" in line.upper()
                or "CORRECT" in line.upper()
                or re.search(r"(?:^|\s)([+-]1)(?:\s|</|$)", line)
            ]
            feedback_summary = (
                "\n".join(feedback_lines) if feedback_lines else particle.feedback
            )
            incorrect_lines = [
                line
                for line in feedback_lines
                if "INCORRECT" in line.upper()
                or re.search(r"(?:^|\s)-1(?:\s|</|$)", line)
            ]
            if incorrect_lines:
                feedback_summary = "\n".join(incorrect_lines)
            noise_instruction = ""
            if random.random() < self.cfg.acceptance_noise:
                noise_instruction = (
                    "\n\nIMPORTANT: Even if the solution seems correct, "
                    "explore a DIFFERENT reasoning approach. Be creative and "
                    "try an alternative method to solve this problem."
                )
            prev_steps = extract_steps(particle.answer.reasoning)
            prev_reasoning = (
                "\n".join(f"<step>{s}</step>" for s in prev_steps) if prev_steps else ""
            )
            transport_prompt = (
                f"Problem:\n{state.question}\n\n"
                f"Previous reasoning:\n{prev_reasoning}\n"
                f"Previous choice: {particle.answer.choice}\n\n"
                f"Step-by-step verification:\n{feedback_summary}\n\n"
                "Refine the solution by fixing any -1 steps. "
                "Maintain +1 steps unless you find a better approach.\n\n"
                "Formatting requirements:\n"
                "- Output ONLY <step>...</step> lines.\n"
                "- Do NOT include markdown headings, separators, or text outside <step> tags.\n"
                "- Keep the number of steps small and concise.\n"
                "- Your LAST step MUST contain the final answer: "
                "<step>The final answer is: \\boxed{...}</step>\n"
                f"{noise_instruction}"
            )
            max_attempts = max(1, self.cfg.max_attempts)
            last_exc: Exception | None = None
            result = None
            for attempt in range(max_attempts):
                try:
                    result = await asyncio.wait_for(
                        state.iterator_agent.run(transport_prompt),
                        timeout=self.cfg.proposal_timeout_s,
                    )
                    break
                except (asyncio.TimeoutError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        transport_prompt = (
                            f"Problem:\n{state.question}\n\n"
                            f"Previous choice: {particle.answer.choice}\n\n"
                            f"Incorrect steps to fix:\n{feedback_summary}\n\n"
                            "Write a corrected solution and the final answer.\n\n"
                            "Formatting requirements:\n"
                            "- Output ONLY <step>...</step> lines.\n"
                            "- Keep it concise.\n"
                        )
                        continue
                    return particle.answer
                except Exception as exc:
                    last_exc = exc
                    if self.cfg.debug:
                        raise
                    if attempt < max_attempts - 1:
                        continue
                    return particle.answer
            if result is None:
                if last_exc is not None and self.cfg.debug:
                    raise last_exc
                return particle.answer
            state.context.record_usage(result.usage())
            proposal = parse_answer(result.output, question_type=state.question_type)
            proposal = coerce_answer_stepwise(proposal)
            proposal_feedback, proposal_score = await self._run_verifier(
                state,
                proposal.reasoning,
                proposal.choice,
            )
            proposal_weight = score_to_weight(proposal_score, state.temperature)
            accepted, _ = acceptance_decision(
                particle.weight,
                proposal_weight,
                proposal_score,
                particle.score,
            )
            if accepted:
                return proposal
            return particle.answer
        except Exception:
            if self.cfg.debug:
                raise
            return particle.answer
__all__ = [
    "Prism",
    "PrismConfig",
    "ArbitrationDecision",
    "ScoredParticle",
    "parse_answer",
    "extract_steps",
    "systematic_resample",
    "compute_ess",
    "acceptance_decision",
    "select_arbitration_candidates",
]
