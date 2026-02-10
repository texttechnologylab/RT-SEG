import copy
from typing import Literal

import pandas as pd
from surrealdb import Surreal
from tqdm import tqdm

from rt_segmentation import sdb_login, evaluate_aggregate_segmentations, evaluate_segmentations, \
    evaluate_approaches_bounding_similarity
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
                             RTZeroShotSeqClassificationRF, import_annotated_data)



def human_agreement():
    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])

        res = db.query("SELECT *, ->has_though_anchor_gold_ha->though_anchor_gold_ha.* as ha, ->has_though_anchor_gold_ve->though_anchor_gold_ve.* as ve from rtrace")

    traces = []
    anno_data = []
    for rtrace in res:
        traces.append(rtrace["rt"])
        anno_data.append({"ve": rtrace["ve"][0]["split"],
                          "ha": rtrace["ha"][0]["split"]})

    """# print(traces[0])
    # print(anno_data[0])

    # Aggregate across multiple traces (here using the same trace 3 times)
    agg_tables = evaluate_aggregate_segmentations(
        traces=traces,
        segmentations=anno_data,
        gold_key="ha",
        sigma=5.0,
        window=3,
        slack=10,
    )

    print("\n=== Aggregated Linear Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['linear_metrics'])

    print("\n=== Aggregated Agreement Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['pairwise_agreement_metrics'])

    print("\n=== Aggregated Agreement Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['per_method_agreement_metrics'])

    # Aggregate across multiple traces (here using the same trace 3 times)
    agg_tables = evaluate_aggregate_segmentations(
        traces=traces,
        segmentations=anno_data,
        gold_key="ve",
        sigma=5.0,
        window=3,
        slack=10,
    )

    print("\n=== Aggregated Linear Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['linear_metrics'])

    print("\n=== Aggregated Agreement Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['pairwise_agreement_metrics'])

    print("\n=== Aggregated Agreement Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['per_method_agreement_metrics'])"""

    evaluate_approaches_bounding_similarity(traces, anno_data)



def score_approaches_triadic_boundary_similarity(human_baseline: Literal["reasoning_flow_gold",
                                                                        "thought_anchor_gold",
                                                                        "comb"] = "thought_anchor_gold"):
    models = [
        # [RTLLMOffsetBased],
        [RTLLMForcedDecoderBased],
        [RTLLMSegUnitBased],
        [RTRuleRegex],
        [RTNewLine],
        [RTPRMBase],
        [RTEntailmentBasedSegmentation],
        [RTLLMEntropy],
        [RTLLMTopKShift],
        [RTLLMFlatnessBreak],
        [RTLLMSurprisal],
       # [RTBERTopicSegmentation],
        [RTZeroShotSeqClassification],
        [RTLLMReasoningFlow],
        [RTLLMArgument],
        [RTLLMThoughtAnchor],
        [RTEmbeddingBasedSemanticShift],
        [RTZeroShotSeqClassificationRF],
        [RTZeroShotSeqClassificationTA],
        [RTZeroShotSeqClassificationRF, RTZeroShotSeqClassificationTA],
        [RTLLMReasoningFlow, RTLLMThoughtAnchor],
        [RTLLMReasoningFlow, RTLLMThoughtAnchor, RTLLMArgument],
        [RTLLMThoughtAnchor, RTZeroShotSeqClassificationTA],
        [RTLLMReasoningFlow, RTZeroShotSeqClassificationRF],
        [RTNewLine, RTRuleRegex, RTLLMSegUnitBased]
    ]
    aligner: OffsetFusion = OffsetFusionGraph
    seg_base_unit: Literal["clause", "sent"] = "clause"

    target_experiments = [RTSeg(engines=m, aligner=aligner, seg_base_unit=seg_base_unit).exp_id for m in models]

    if human_baseline == "reasoning_flow_gold":
        gold_keys = ["reasoning_flow_gold"]
    elif human_baseline == "thought_anchor_gold":
        gold_keys = ["thought_anchor_gold_ve",
                     "though_anchor_gold_ve",
                     "thought_anchor_gold_ha",
                     "though_anchor_gold_ha"]
    else:
        gold_keys = ["thought_anchor_gold_ve",
                     "though_anchor_gold_ve",
                     "thought_anchor_gold_ha",
                     "though_anchor_gold_ha",
                     "reasoning_flow_gold"]
    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])

        res = db.query(
            "SELECT *, ->?->?.* from rtrace")

    traces = []
    human_anno_data = dict()
    model_anno_data = dict()
    for rtrace in tqdm(res, desc="Gathering data"):
        traces.append(rtrace["rt"])
        for anno in rtrace["->?"]["->?"]:
            if anno.get("id").table_name in gold_keys:
                if anno.get("id").table_name in human_anno_data:
                    human_anno_data[anno.get("id").table_name].append(anno["split"])
                else:
                    human_anno_data[anno.get("id").table_name] = [anno["split"]]
            elif anno.get("id").table_name in target_experiments:
                if anno.get("id").table_name in model_anno_data:
                    model_anno_data[anno.get("id").table_name].append(anno["split"])
                else:
                    model_anno_data[anno.get("id").table_name] = [anno["split"]]
            else:
                pass

    print(*human_anno_data.items(), sep="\n")
    print(*model_anno_data.items(), sep="\n")

    assert len(list(set([len(v) for (k, v) in human_anno_data.items()]))) == 1, [(k, len(v)) for (k, v) in human_anno_data.items()]
    assert len(list(set([len(v) for (k, v) in model_anno_data.items()]))) == 1, [(k, len(v)) for (k, v) in model_anno_data.items()]
    assert set([len(v) for (k, v) in human_anno_data.items()]) == set([len(v) for (k, v) in model_anno_data.items()])

    for model in model_anno_data:
        current_data = copy.deepcopy(human_anno_data)
        current_data[model] = model_anno_data[model]
        target_data = []
        for idx in range(len(current_data[[*current_data.keys()][0]])):
            target_data.append({k: v[idx] for (k, v) in current_data.items()})

        score = evaluate_approaches_bounding_similarity(traces, target_data)
        print(f"{model} group score: {score:.3f}")

if __name__ == "__main__":
    score_approaches_triadic_boundary_similarity()