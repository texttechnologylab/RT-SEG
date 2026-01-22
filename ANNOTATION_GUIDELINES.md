# Reasoning Trace Segmentation Guidelines

## 1. Task Overview

You are given a *reasoning trace* produced by a language model.  
Your task is to segment the trace into **reasoning segments** by inserting boundaries **between sentences**.

Each segment should correspond to **one coherent reasoning step**.

You do **not** need to judge correctness, logic, or factual accuracy.

---

## 2. Definition of a Reasoning Segment

> A **reasoning segment** is a contiguous span of text that performs a *single epistemic function* in the reasoning process.

An epistemic function describes *what role* a span plays in the reasoning (e.g., setting up assumptions, deriving a consequence, revising a claim).

---

## 3. When to Insert a Boundary

Insert a boundary **between two sentences** if the later sentence introduces a **new epistemic function**.

Boundaries may occur even if:
- [the topic stays the same] ?
- the same entities are discussed
- similar vocabulary is reused

---

## 4. Epistemic Functions (Reference Inventory)

Use the following functions as guidance for identifying boundaries.  
You **do not need to label them**, only decide whether the function changes.

1. **Assumption / Setup**  
   Introducing premises, constraints, or known facts.

2. **Intermediate Inference**  
   Deriving new information from prior content.

3. **Contrast / Objection**  
   Introducing alternatives, counterarguments, or limitations.

4. **Revision / Correction**  
   Retracting, correcting, or reconsidering earlier reasoning.

5. **Goal Management**  
   Stating subgoals, plans, or next reasoning steps.

6. **Conclusion / Answer**  
   Presenting a final result or summarizing outcome.

---

## 5. Hard Constraints

- Boundaries may only be placed **between sentences**
- **Do not split within a sentence**
- Segments must be **contiguous and non-overlapping**
- Every character in the text must belong to exactly one segment

---

## 6. Granularity Rules

- [**Do not split** minor paraphrases or restatements] ?
- **Do not split** arithmetic or symbolic chains unless interrupted
- **Do split** when the *role* of the text changes, even subtly
- When in doubt, **do not split**

---

## 7. Canonical Examples

### Example 1: Inference shift (split)

