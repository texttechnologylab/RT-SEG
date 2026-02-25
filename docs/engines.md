---
layout: default
title: Engines
nav_order: 4
has_children: true
---

# Available Engines

## Rule-Based

* `RTRuleRegex`
* `RTNewLine`

## LLM-Based (Boundary Inference)

* `RTLLMOffsetBased`
* `RTLLMSegUnitBased`
* `RTLLMForcedDecoderBased`
* `RTLLMSurprisal`
* `RTLLMEntropy`
* `RTLLMTopKShift`
* `RTLLMFlatnessBreak`

## Discourse / Reasoning Schemas

* `RTLLMThoughtAnchor`
* `RTLLMReasoningFlow`
* `RTLLMArgument`

## Topic / Semantic / NLI

* `RTBERTopicSegmentation`
* `RTEmbeddingBasedSemanticShift`
* `RTEntailmentBasedSegmentation`
* `RTZeroShotSeqClassification`
* `RTZeroShotSeqClassificationRF`
* `RTZeroShotSeqClassificationTA`

## PRM-Based

* `RTPRMBase`

---

# Engine Configuration

You can override engine parameters at call time:

```python
from rt_seg import RTSeg
from rt_seg import RTRuleRegex
from rt_seg import RTBERTopicSegmentation
from rt_seg import OffsetFusionGraph

segmentor = RTSeg(
    engines=[RTRuleRegex, RTBERTopicSegmentation],
    aligner=OffsetFusionGraph,
    label_fusion_type="concat",  # or "majority"
    seg_base_unit="clause",
)
trace = "..."
offsets, labels = segmentor(
    trace,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    chunk_size=200,
)
```