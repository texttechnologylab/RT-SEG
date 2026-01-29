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

Insert a boundary **between two sentences** whenever a reasoning step ends. A step ends if:

1. The epistemic function **changes** (e.g., Framing → Inference → Pivot), **or**
2. The current reasoning step is **complete**, even if the function remains the same.

> Note: This ensures that repeated reasoning steps of the same function (e.g., multiple consecutive Inference steps) are treated as separate segments.

Boundaries may occur even if:
- The topic stays the same
- The same entities are discussed
- Similar vocabulary is reused

**Example:**

```

Sentence 1: Assume x > 0.       → Framing (segment 1)
Sentence 2: Assume y > 0.       → Framing (segment 2, new segment)
Sentence 3: Therefore x + y > 0 → Inference (segment 3, new segment)

```

---

## 4. Epistemic Functions (Reference Inventory)

Use the following functions as guidance for identifying boundaries.  
You **do not need to label them**, only decide whether a new reasoning step has begun.

| Function | Description | Clues / Keywords |
|----------|------------|----------------|
| **1. Framing** | Setting the stage: restating the goal, defining variables, or listing known constraints. | "Let $x$ be...", "We need to find...", "The rules are..." |
| **2. Inference** | The proactive "forward" step: deriving new information, performing math, or narrowing the search space. | "Therefore...", "This means...", "If A, then B..." |
| **3. Pivot** | A change in direction: includes self-correction, identifying errors, or switching strategies. | "Wait...", "Actually...", "Alternatively...", "On the other hand..." |
| **4. Verification** | Checking the work: not deriving new info, but validating that the current path/result is consistent. | "Let's double check...", "This matches the constraint...", "Testing this value..." |
| **5. Conclusion** | The final landing: explicitly stating the answer or a final summary. | "So, the final answer is...", "In conclusion..." |

---

## 5. Hard Constraints

- Boundaries may only be placed **between sentences**
- **Do not split within a sentence**
- Segments must be **contiguous and non-overlapping**
- Every character in the text must belong to exactly one segment

---

## 6. Granularity Rules

- **Do not split** minor paraphrases or restatements
- **Do not split** arithmetic or symbolic chains unless interrupted by explanation, verification, or pivot
- **Do split** when the *role* of the text changes, or when a reasoning step ends, even if the function remains the same
- When in doubt, **do not split**

---

## 7. Edge Cases

- **Consecutive steps of the same function:** Each step counts as a new segment (e.g., multiple inferences in a row are separate)
- **Meta-comments:** Statements like "Let’s think carefully" start a new segment only if they indicate a goal or strategy change
- **Repetition:** Rephrasing or restating a previous step does **not** create a new segment
- **Arithmetic or symbolic reasoning chains:** Treat as one segment unless interrupted by verification, pivot, or explanation

---

## 8. What Not to Annotate

- Logical validity
- Factual correctness
- Implicit or unstated reasoning
- How a human *should* reason

Focus only on the **epistemic role of the explicit text**.

## 9. Literature Examples
Overall:
- discourse segmentation
- argumentation / reasoning structure
- cognitive models of problem solving

### Discourse Segmentation

Justify:
- sentence level segmentation
- functional shifts
- non-topical boundaries

#### Example 1: Mann & Thompson (1988) — RST (Rethorical Structure Theory)
Defines:
- Introduces discourse units defined by function, not topic.
```bibtex
@article{MANNTHOMPSON+1988+243+281,
    url = {https://doi.org/10.1515/text.1.1988.8.3.243},
    title = {Rhetorical Structure Theory: Toward a functional theory of text organization},
    title = {},
    author = {WILLIAM C. MANN and SANDRA A. THOMPSON},
    pages = {243--281},
    volume = {8},
    number = {3},
    journal = {Text - Interdisciplinary Journal for the Study of Discourse},
    doi = {doi:10.1515/text.1.1988.8.3.243},
    year = {1988},
    lastchecked = {2026-01-22}
}
```

#### Example 2: Carlson, Marcu & Okurowski (2001) — Building a Discourse-Tagged Corpus in the Framework of Rhetorical Structure Theory
Defines:
- Defines Elementary Discourse Units (EDUs), segmented conservatively.
```bibtex
@inproceedings{carlson-etal-2001-building,
    title = "Building a Discourse-Tagged Corpus in the Framework of {R}hetorical {S}tructure {T}heory",
    author = "Carlson, Lynn  and
      Marcu, Daniel  and
      Okurovsky, Mary Ellen",
    booktitle = "Proceedings of the Second {SIG}dial Workshop on Discourse and Dialogue",
    year = "2001",
    url = "https://aclanthology.org/W01-1605/"
}
```

### Argumentation & reasoning structure (your conceptual foundation)

Justify:
- inference vs. premise vs. conclusion
- revision and objection as first-class units
- 
#### Example 1: Toulmin (1958) - The Uses of Argument
Defines:
- Classic model separating claims, grounds, warrants (inference segments, contrast/objection, conclusion)
```bibtex
@book{toulmin2003uses,
  title={The uses of argument},
  author={Toulmin, Stephen E},
  year={2003},
  publisher={Cambridge university press}
}
```
and
```bibtex
@article{karbach1987using,
  title={Using Toulmin's model of argumentation},
  author={Karbach, Joan},
  journal={Journal of Teaching Writing},
  volume={6},
  number={1},
  pages={81--92},
  year={1987}
}
```
#### Example 2: Stede (2011) — Argumentation mining
Defines:
- Explicitly argues that argument units are discourse-level spans, not logical steps.
```bibtex
@book{stede2019argumentation,
  title={Argumentation mining},
  author={Stede, Manfred and Schneider, Jodi and Hirst, Graeme},
  year={2019},
  publisher={Springer}
}
```
#### Example 3: Argument Mining Overview
Defines:
- Overview of argument mining literature, might be helpful?
```bibtex
@article{lawrence-reed-2019-argument,
    title = "Argument Mining: A Survey",
    author = "Lawrence, John  and
      Reed, Chris",
    journal = "Computational Linguistics",
    volume = "45",
    number = "4",
    month = dec,
    year = "2019",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/J19-4006/",
    doi = "10.1162/coli_a_00364",
    pages = "765--818",
    abstract = "Argument mining is the automatic identification and extraction of the structure of inference and reasoning expressed as arguments presented in natural language. Understanding argumentative structure makes it possible to determine not only what positions people are adopting, but also why they hold the opinions they do, providing valuable insights in domains as diverse as financial market prediction and public relations. This survey explores the techniques that establish the foundations for argument mining, provides a review of recent advances in argument mining techniques, and discusses the challenges faced in automatically extracting a deeper understanding of reasoning expressed in language in general."
}
```

### Reasoning, problem solving & cognitive steps

Justify:
- “epistemic function” language

#### Example 1: Newell & Simon (1971) — Human problem solving
Defines:
- Reasoning proceeds via state transitions, not atomic inferences (goal management, revision, multi.sentence steps)
```bibtex
@article{simon1971human,
  title={Human problem solving: The state of the theory in 1970.},
  author={Simon, Herbert A and Newell, Allen},
  journal={American psychologist},
  volume={26},
  number={2},
  pages={145},
  year={1971},
  publisher={American Psychological Association}
}
```
#### Example 2: Polya (1945) — How to Solve It
Defines:
- Defines reasoning in phases: understand, plan, execute, review
```bibtex
@article{polya1957solve,
  title={How to solve it},
  author={Polya, George},
  year={1957},
  publisher={Princeton Press}
}
```

### Recent LLM / reasoning trace–adjacent work

Justify:
- Introduces reasoning traces as explicit text objects, and suggests segmentations etc.

#### Example 1: Wei et al. (2022) — Chain-of-Thought
Defines:
- Introduces CoT traces
```bibtex
@misc{wei2023chainofthoughtpromptingelicitsreasoning,
      title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models}, 
      author={Jason Wei and Xuezhi Wang and Dale Schuurmans and Maarten Bosma and Brian Ichter and Fei Xia and Ed Chi and Quoc Le and Denny Zhou},
      year={2023},
      eprint={2201.11903},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2201.11903}, 
}
```

#### Example 2: REASONINGFLOW: Semantic Structure of Complex Reasoning Traces
Defines:
- Introduces flow based segmentation with labels and edges (text segments as nodes in DAG)
```bibtex
@misc{lee2025reasoningflowsemanticstructurecomplex,
      title={ReasoningFlow: Semantic Structure of Complex Reasoning Traces}, 
      author={Jinu Lee and Sagnik Mukherjee and Dilek Hakkani-Tur and Julia Hockenmaier},
      year={2025},
      eprint={2506.02532},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.02532}, 
}
```

#### Example 3: The Geometry of Reasoning: Flowing Logics in Representation Space
Defines:
- Introduces flow based segmentation with labels
```bibtex
@misc{zhou2025geometryreasoningflowinglogics,
      title={The Geometry of Reasoning: Flowing Logics in Representation Space}, 
      author={Yufa Zhou and Yixiao Wang and Xunjian Yin and Shuyan Zhou and Anru R. Zhang},
      year={2025},
      eprint={2510.09782},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.09782}, 
}
```
---

## 8. Canonical Examples

### Example 1: Inference shift (split)

