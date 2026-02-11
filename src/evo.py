import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import random
import os
import time
from functools import lru_cache
from typing import List, Any, Tuple, Optional, Dict

from surrealdb import Surreal

from rt_segmentation import RTSeg, OffsetFusionFuzzy, OffsetFusionIntersect, OffsetFusionMerge
from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSegUnitBased,
                             RTRuleRegex,
                             RTNewLine,
                             RTPRMBase,
                             RTLLMReasoningFlow,
                             RTLLMArgument,
                             RTLLMThoughtAnchor,
                             RTLLMEntropy,
                             RTLLMTopKShift,
                             RTLLMFlatnessBreak,
                             RTLLMSurprisal,
                             RTBERTopicSegmentation,
                             RTZeroShotSeqClassification,
                             RTEntailmentBasedSegmentation,
                             RTEmbeddingBasedSemanticShift,
                             bp, sdb_login, load_prompt, load_example_trace,
                             export_gold_set, export_rf_data_gold_set, upload_rf_data,
                             OffsetFusionGraph,
                             RTSeg,
                             OffsetFusion,
                             RTZeroShotSeqClassificationTA,
                             RTZeroShotSeqClassificationRF,
                             import_annotated_data,
                             OffsetFusionVoting,
                             score_approaches_triadic_boundary_similarity_one_model)

# ----------------------------- USER TO FILL THESE -----------------------------
# List your 20 model engines here (in a fixed order - the GA will always select/subset them in this order)
ALL_MODELS: List[Any] = [
        RTLLMOffsetBased,
        RTLLMForcedDecoderBased,
        RTLLMSegUnitBased,
        RTRuleRegex,
        RTNewLine,
        RTPRMBase,
        RTEntailmentBasedSegmentation,
        RTLLMEntropy,
        RTLLMTopKShift,
        RTLLMFlatnessBreak,
        RTLLMSurprisal,
        RTBERTopicSegmentation,
        RTZeroShotSeqClassification,
        RTLLMReasoningFlow,
        RTLLMArgument,
        RTLLMThoughtAnchor,
        RTEmbeddingBasedSemanticShift,
        RTZeroShotSeqClassificationRF,
        RTZeroShotSeqClassificationTA,
    ]

# List your 6 possible aligner objects here
ALIGNERS: List[Any] = [OffsetFusionGraph, OffsetFusionFuzzy, OffsetFusionIntersect, OffsetFusionMerge, OffsetFusionVoting]





def run_single_model_exp(model_list: List[Any], aligner: Optional[Any], seg_base_unit: bool) -> str:
    """
    Runs segmentation and returns exp_id (string). Raises on failure.

    NOTE: model_list elements and aligner are UNINITIALIZED symbols (usually classes).
    """
    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])

        tables = [*db.query("info for DB").get("tables").keys()]

    if seg_base_unit:
        seg_base_unit = "clause"
    else:
        seg_base_unit = "sent"
    rt_seg = RTSeg(
        engines=model_list,
        aligner=None if len(model_list) == 1 else aligner,
        label_fusion_type="concat",
        seg_base_unit=seg_base_unit,
    )

    if rt_seg.exp_id in tables:
        return rt_seg.exp_id

    rt_seg.sdb_segment_ds(
        exp_id=rt_seg.exp_id,
        clear=True,
        db="RT_RF",
        seg_base_unit=seg_base_unit,
    )
    return rt_seg.exp_id


# ----------------------------- GA hyperparameters -----------------------------
NUM_MODELS = len(ALL_MODELS)
NUM_ALIGNERS = len(ALIGNERS)

POP_SIZE = 24
NUM_GENERATIONS = 20

MIN_MODELS_PER_INDIVIDUAL = 1
MAX_MODELS_PER_INDIVIDUAL = 3

MUTATION_RATE = 0.02
ALIGNER_MUTATION_PROB = 0.2
SEG_MUTATION_PROB = 0.2
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 4

NUM_PROCESSES = 6

TIME_BUDGET_HOURS = 48
TIME_BUDGET_SECONDS = TIME_BUDGET_HOURS * 3600
MAX_UNIQUE_EVALS = int(144 * NUM_PROCESSES)  # ~20 min per eval, ideal utilization

Individual = Tuple[int, int, int]


def repair_mask(mmm: int) -> int:
    num = mmm.bit_count()

    # Too many models → randomly turn some off
    if num > MAX_MODELS_PER_INDIVIDUAL:
        ones = [i for i in range(NUM_MODELS) if mmm & (1 << i)]
        to_remove = random.sample(ones, num - MAX_MODELS_PER_INDIVIDUAL)
        for i in to_remove:
            mmm ^= (1 << i)

    # Too few models → randomly turn some on
    elif num < MIN_MODELS_PER_INDIVIDUAL:
        zeros = [i for i in range(NUM_MODELS) if not (mmm & (1 << i))]
        to_add = random.sample(zeros, MIN_MODELS_PER_INDIVIDUAL - num)
        for i in to_add:
            mmm |= (1 << i)

    return mmm

def _validate_search_space() -> None:
    if NUM_MODELS <= 0:
        raise ValueError("ALL_MODELS is empty. Fill it with model classes.")
    if NUM_ALIGNERS <= 0:
        raise ValueError("ALIGNERS is empty. Fill it with aligner classes.")
    if not (1 <= MIN_MODELS_PER_INDIVIDUAL <= MAX_MODELS_PER_INDIVIDUAL):
        raise ValueError("Invalid MIN/MAX models per individual constraints.")
    if POP_SIZE < 2:
        raise ValueError("POP_SIZE must be >= 2.")
    if TOURNAMENT_SIZE < 2:
        raise ValueError("TOURNAMENT_SIZE must be >= 2.")
    if TOURNAMENT_SIZE > POP_SIZE:
        raise ValueError("TOURNAMENT_SIZE cannot exceed POP_SIZE.")


def random_individual() -> Individual:
    while True:
        mask = random.randint(0, (1 << NUM_MODELS) - 1)
        num_selected = mask.bit_count()
        if MIN_MODELS_PER_INDIVIDUAL <= num_selected <= MAX_MODELS_PER_INDIVIDUAL:
            break
    aligner_idx = random.randint(0, NUM_ALIGNERS - 1)
    seg_base_unit = random.randint(0, 1)
    return mask, aligner_idx, seg_base_unit


def _decode_to_runtime_objects(ind: Individual) -> Tuple[List[Any], Optional[Any], bool]:
    mask, aligner_idx, seg_int = ind

    model_list = [ALL_MODELS[i] for i in range(NUM_MODELS) if (mask & (1 << i))]
    if len(model_list) <= 1:
        aligner = None
    else:
        aligner = ALIGNERS[aligner_idx]

    seg_base_unit = bool(seg_int)
    return model_list, aligner, seg_base_unit


def _fitness_worker(ind: Individual) -> float:
    try:
        model_list, aligner, seg_base_unit = _decode_to_runtime_objects(ind)
        exp_id = run_single_model_exp(model_list, aligner, seg_base_unit)
        return float(score_approaches_triadic_boundary_similarity_one_model(exp_id))
    except Exception as e:
        print(f"Worker failed for {ind}: {e}")
        return 0.0


def tournament_selection(pop_with_fitness: List[Tuple[Individual, float]], k: int = TOURNAMENT_SIZE) -> Individual:
    k = min(k, len(pop_with_fitness))
    candidates = random.sample(pop_with_fitness, k)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    mask1, a1, s1 = parent1
    mask2, a2, s2 = parent2

    child_mask1 = 0
    child_mask2 = 0
    for i in range(NUM_MODELS):
        if random.random() < 0.5:
            child_mask1 |= mask1 & (1 << i)
            child_mask2 |= mask2 & (1 << i)
        else:
            child_mask1 |= mask2 & (1 << i)
            child_mask2 |= mask1 & (1 << i)

    child_mask1 = repair_mask(child_mask1)
    child_mask2 = repair_mask(child_mask2)

    child_a1, child_a2 = (a1, a2) if random.random() < 0.5 else (a2, a1)
    child_s1, child_s2 = (s1, s2) if random.random() < 0.5 else (s2, s1)

    return (child_mask1, child_a1, child_s1), (child_mask2, child_a2, child_s2)


def mutate(ind: Individual) -> Individual:
    mask, aligner_idx, seg = ind

    for i in range(NUM_MODELS):
        if random.random() < MUTATION_RATE:
            mask ^= (1 << i)

    mask = repair_mask(mask)

    if random.random() < ALIGNER_MUTATION_PROB:
        aligner_idx = random.randint(0, NUM_ALIGNERS - 1)

    if random.random() < SEG_MUTATION_PROB:
        seg = 1 - seg

    return mask, aligner_idx, seg


if __name__ == "__main__":
    _validate_search_space()
    random.seed(42)

    start_time = time.time()

    fitness_cache: Dict[Individual, float] = {}
    unique_evals = 0

    population: List[Individual] = [random_individual() for _ in range(POP_SIZE)]
    best_individual: Optional[Individual] = None
    best_score: float = -float("inf")

    with mp.Pool(
            processes=NUM_PROCESSES,
            maxtasksperchild=1
    ) as pool:
        for generation in range(NUM_GENERATIONS):
            elapsed = time.time() - start_time
            if elapsed >= TIME_BUDGET_SECONDS:
                print(f"Time budget reached ({elapsed/3600:.2f}h). Stopping early.")
                break
            if unique_evals >= MAX_UNIQUE_EVALS:
                print(f"Evaluation budget reached ({unique_evals} unique evals). Stopping early.")
                break

            unique_population = list(dict.fromkeys(population))
            to_eval = [ind for ind in unique_population if ind not in fitness_cache]

            if to_eval:
                remaining = MAX_UNIQUE_EVALS - unique_evals
                to_eval = to_eval[: max(0, remaining)]
                new_scores = pool.map(_fitness_worker, to_eval)
                for ind, sc in zip(to_eval, new_scores):
                    fitness_cache[ind] = sc
                unique_evals += len(to_eval)

            fitness_scores = [fitness_cache[ind] for ind in population]
            pop_with_fitness = list(zip(population, fitness_scores))

            current_best = max(pop_with_fitness, key=lambda x: x[1])
            if current_best[1] > best_score:
                best_score = current_best[1]
                best_individual = current_best[0]

            avg_models = sum(ind[0].bit_count() for ind in population) / POP_SIZE
            print(
                f"Gen {generation:3d} | Best gen: {current_best[1]:.4f} | "
                f"Global best: {best_score:.4f} | Avg models: {avg_models:.1f} | "
                f"Unique evals: {unique_evals}/{MAX_UNIQUE_EVALS} | "
                f"Elapsed: {elapsed/3600:.2f}h"
            )

            pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
            new_population: List[Individual] = [ind for ind, _ in pop_with_fitness[:2]]

            while len(new_population) < POP_SIZE:
                parent1 = tournament_selection(pop_with_fitness)
                parent2 = tournament_selection(pop_with_fitness)

                if random.random() < CROSSOVER_RATE:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                new_population.append(mutate(child1))
                if len(new_population) < POP_SIZE:
                    new_population.append(mutate(child2))

            population = new_population

    print("\n=== EVOLUTIONARY SEARCH COMPLETE ===")
    print(f"Best score found: {best_score:.6f}")
    if best_individual is not None:
        mask, aligner_idx, seg_int = best_individual
        selected_indices = [i for i in range(NUM_MODELS) if (mask & (1 << i))]
        aligner_name = ALIGNERS[aligner_idx].__name__ if len(selected_indices) > 1 else "None (single model)"
        print("Best configuration:")
        print(f"  Selected model indices: {selected_indices} (count = {len(selected_indices)})")
        print(f"  Aligner: {aligner_name}")
        print(f"  seg_base_unit: {bool(seg_int)}")