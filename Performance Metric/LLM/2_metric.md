
---

### 1. F1 (QA/Task-specific)

**Formula:**
<img width="583" height="166" alt="image" src="https://github.com/user-attachments/assets/54ba9ef3-6dec-441d-ac9a-108e55532e3c" />

- TP: True Positives (correctly predicted tokens, spans, or answers).
- FP: False Positives (incorrectly predicted tokens, spans, or answers).
- FN: False Negatives (missed tokens, spans, or answers).

**Explanation:**
In question answering (QA) or task-specific contexts, F1 measures the balance between precision and recall for predicted answers, typically at the token or span level. For QA, it evaluates the overlap between predicted and reference answer spans (e.g., in SQuAD). In other tasks like named entity recognition (NER), it assesses correctly identified entities.

**When, Where, and Why It Is Used:**
- **When**: Used in QA (e.g., extractive QA like SQuAD) or tasks requiring precise span or token identification (e.g., NER, slot filling).
- **Where**: Common in datasets like SQuAD, CoQA, or NER datasets (e.g., CoNLL-2003).
- **Why**: F1 balances precision (avoiding incorrect predictions) and recall (capturing all correct answers), providing a single metric for tasks with partial matches.
- **Role**: Standard metric for evaluating answer span accuracy in QA or entity identification in task-specific settings, complementing stricter metrics like Exact Match (EM).

**Advantages:**
- Balances precision and recall, rewarding partial correctness.
- Flexible for token-level, span-level, or entity-level evaluation.
- Widely used, enabling comparison across models.

**Limitations:**
- Does not capture semantic similarity (e.g., synonyms or paraphrases).
- May require strict or relaxed matching criteria, affecting results.
- Less informative for tasks where exact matches are critical (e.g., use EM instead).

**Example Use Case:**
In extractive QA on SQuAD, F1 evaluates how well a model’s predicted answer span (e.g., “Joe Biden”) overlaps with the reference answer (e.g., “President Joe Biden”) for the question “Who is the president of the USA?” (as of August 15, 2025).

---

### 2. HumanEval (Pass@k)

**Formula:**
<img width="707" height="299" alt="image" src="https://github.com/user-attachments/assets/7ac7f844-8263-423d-9e3c-38155e0c28e1" />


**Explanation:**
Pass@k measures the probability that at least one of the top \( k \) generated solutions (e.g., code snippets) for a programming problem passes all test cases. It is computed by sampling \( n \) solutions, counting how many are correct (\( c \)), and calculating the probability of selecting at least one correct solution in \( k \) attempts.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate code generation models.
- **Where**: Standard in datasets like HumanEval, which contains 164 Python programming problems with test cases.
- **Why**: Pass@k accounts for the stochastic nature of code generation, rewarding models that produce at least one correct solution within \( k \) attempts.
- **Role**: Primary metric for assessing functional correctness of generated code, widely used in evaluating large language models (LLMs) for programming tasks.

**Advantages:**
- Directly measures functional correctness via test cases.
- Accounts for multiple solution attempts, reflecting real-world usage.
- Robust to variations in code style, as long as the solution passes tests.

**Limitations:**
- Requires well-designed test cases, which may miss edge cases.
- Does not evaluate code quality (e.g., readability, efficiency).
- Dependent on the number of samples (\( n \)) and \( k \).

**Example Use Case:**
In evaluating a code generation model on HumanEval, Pass@1 measures the percentage of problems where the model’s first generated Python function passes all test cases (e.g., a function to compute factorial).

---

### 3. HellaSwag Accuracy

**Formula:**
<img width="660" height="108" alt="image" src="https://github.com/user-attachments/assets/d7b13903-04b7-4418-8215-1f1e5c0c2f49" />

- Correct prediction: The model selects the correct ending for a given context from multiple-choice options.

**Explanation:**
HellaSwag Accuracy evaluates a model’s ability to perform commonsense reasoning by selecting the most plausible ending for a partially completed story or scenario from a set of multiple-choice options. The HellaSwag dataset tests machine understanding of real-world scenarios.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate commonsense reasoning in language models.
- **Where**: Specific to the HellaSwag dataset, which includes contexts from sources like ActivityNet and WikiHow.
- **Why**: HellaSwag challenges models with adversarial endings that are plausible but incorrect, testing deep understanding of context and commonsense.
- **Role**: Assesses a model’s ability to understand nuanced, human-like reasoning in open-domain scenarios.

**Advantages:**
- Tests complex commonsense reasoning, a key aspect of human-like understanding.
- Simple to compute as a multiple-choice accuracy metric.
- Challenging dataset, pushing models beyond simple pattern matching.

**Limitations:**
- Limited to multiple-choice format, which may not reflect open-ended reasoning.
- Dataset-specific, limiting generalizability.
- Performance depends on model’s exposure to similar contexts during training.

**Example Use Case:**
In HellaSwag, a model is given a context (e.g., “A person is baking a cake and takes it out of the oven…”) and must choose the correct ending (e.g., “It smells delicious”) over incorrect ones (e.g., “It starts singing”). Accuracy measures correct selections.

---

### 4. MMLU (Massive Multitask Language Understanding)

**Formula:**
<img width="769" height="106" alt="image" src="https://github.com/user-attachments/assets/0392cf6e-fcf3-4b25-8f33-b3857063c8af" />

- $\( T \)$ : Number of tasks (57 in MMLU).
- Accuracy is averaged across all tasks, typically in a multiple-choice format.

**Explanation:**
MMLU evaluates a model’s general knowledge and reasoning across 57 tasks covering STEM, humanities, social sciences, and professional fields (e.g., medicine, law). Each task is a multiple-choice question set, and the overall score is the average accuracy across tasks.

**When, Where, and Why It Is Used:**
- **When**: Used to assess a model’s broad knowledge and reasoning capabilities.
- **Where**: Standard on the MMLU dataset, designed for evaluating large language models.
- **Why**: MMLU tests multitask generalization, covering high-school to professional-level questions, making it a comprehensive benchmark for LLMs.
- **Role**: Serves as a standard for comparing LLMs on general-purpose understanding, reflecting their ability to handle diverse domains.

**Advantages:**
- Broad coverage across domains, testing generalization.
- Standardized, enabling comparison across models.
- Includes varying difficulty levels, from beginner to expert.

**Limitations:**
- Limited to multiple-choice format, not testing open-ended reasoning.
- May not capture practical application or contextual nuances.
- Performance depends on training data coverage of MMLU topics.

**Example Use Case:**
In evaluating an LLM on MMLU, the model answers questions like “What is the capital of France?” (high school) or “What is the diagnosis for symptom X?” (medical), with accuracy averaged across all 57 tasks.

---

### 5. TruthfulQA Accuracy

**Formula:**
<img width="772" height="110" alt="image" src="https://github.com/user-attachments/assets/e55ea9b4-82aa-4297-9617-1234c7040b74" />

- Correct and truthful: The model’s answer matches the ground truth and avoids misinformation or hallucinations.

**Explanation:**
TruthfulQA evaluates a model’s ability to provide accurate and truthful answers to questions, particularly those prone to eliciting misconceptions or falsehoods. The dataset includes questions designed to test common human biases or myths.

**When, Where, and Why It Is Used:**
- **When**: Used to assess the truthfulness and factual accuracy of language models.
- **Where**: Specific to the TruthfulQA dataset, covering topics like health, law, and myths.
- **Why**: Ensures models avoid generating misleading or false information, a critical aspect for real-world applications.
- **Role**: Measures a model’s reliability in providing factually correct answers, especially under challenging or ambiguous questions.

**Advantages:**
- Directly tests truthfulness, addressing a key ethical concern.
- Challenges models on common misconceptions, revealing weaknesses.
- Simple to compute as an accuracy metric.

**Limitations:**
- Limited to specific question set, not covering all domains.
- Subjective definition of “truth” in some cases.
- May not capture nuanced or context-dependent truths.

**Example Use Case:**
In TruthfulQA, a model is asked, “Does the moon cause tides?” and must answer correctly (“Yes”) without falling for misconceptions. Accuracy measures the proportion of truthful responses.

---

### 6. Toxicity Score

**Formula:**
No standard formula; typically:
<img width="513" height="93" alt="image" src="https://github.com/user-attachments/assets/f15b4419-f3df-4ec5-a713-c9fdafc5649c" />

- $\( \text{Toxicity}(x_i) \)$: Toxicity probability of text $\( x_i \)$, often computed using a classifier (e.g., Perspective API or a fine-tuned model).
- Scores range from 0 (non-toxic) to 1 (highly toxic).

**Explanation:**
Toxicity Score measures the likelihood that generated text contains harmful, offensive, or inappropriate content, using automated classifiers trained on labeled datasets. Lower scores indicate safer outputs.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate the safety and appropriateness of generated text.
- **Where**: Common in dialogue systems, text generation, and content moderation (e.g., evaluated with tools like Perspective API).
- **Why**: Ensures models produce safe and ethical outputs, critical for user-facing applications.
- **Role**: Assesses the ethical quality of generated text, identifying harmful content like hate speech or insults.

**Advantages:**
- Quantifies safety, a key concern for real-world deployment.
- Automated, allowing large-scale evaluation.
- Can be integrated into model training or filtering.

**Limitations:**
- Classifier-dependent, with potential biases in toxicity definitions.
- Subjective nature of toxicity (cultural or contextual differences).
- May miss subtle or context-specific harmful content.

**Example Use Case:**
In evaluating a chatbot, Toxicity Score uses a classifier to assess whether generated responses (e.g., “You’re an idiot”) are harmful, aiming for low toxicity scores.

---

### 7. Diversity (Distinct-n, Self-BLEU)

**Formulas:**
<img width="862" height="512" alt="image" src="https://github.com/user-attachments/assets/18dd11e7-0f31-478c-abaf-aebe5fb5db39" />


**Explanation:**
Distinct-n measures the diversity of generated text by calculating the proportion of unique n-grams, capturing lexical variety. Self-BLEU measures diversity by computing the BLEU score of each generated text against others, with lower scores indicating more diverse outputs.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate the diversity of text generated by models like GANs, VAEs, or LLMs.
- **Where**: Common in text generation tasks (e.g., story generation, dialogue) and datasets like E2E NLG or CommonGen.
- **Why**: Diversity ensures models generate varied outputs, avoiding repetitive or monotonous text, which is critical for creative applications.
- **Role**: Quantifies the variety of generated text, complementing quality metrics like BLEU or ROUGE.

**Advantages:**
- Distinct-n is simple and directly measures lexical diversity.
- Self-BLEU leverages existing BLEU infrastructure for diversity evaluation.
- Both highlight mode collapse or lack of variety in generation.

**Limitations:**
- Distinct-n does not capture semantic diversity (e.g., different words with similar meanings).
- Self-BLEU is sensitive to BLEU’s limitations (e.g., exact n-gram matching).
- Both may not reflect human-perceived diversity or coherence.

**Example Use Case:**
In evaluating a story generation model, Distinct-2 measures the variety of bigrams in generated stories, while Self-BLEU ensures stories differ significantly from each other.

---

### 8. Coherence Score

**Formula:**
No standard formula; often computed as:
<img width="510" height="110" alt="image" src="https://github.com/user-attachments/assets/42dd91cd-59aa-4c14-b603-af5ea4a4806a" />

- $\( \text{Coherence}(x_i) \)$: Coherence of text $\( x_i \)$ , typically measured using:
  - **Automated methods**: Perplexity of a language model, cosine similarity of sentence embeddings, or discourse relation classifiers.
  - **Human evaluation**: Human ratings of logical flow and consistency (0 to 1 or Likert scale).

**Explanation:**
Coherence Score evaluates how logically and consistently a generated text flows, ensuring that sentences or paragraphs connect meaningfully. It can be computed automatically (e.g., using embeddings or language models) or via human judgments.

**When, Where, and Why It Is Used:**
- **When**: Used to assess the logical flow of generated text in tasks like story generation, dialogue, or summarization.
- **Where**: Common in open-ended generation tasks or datasets like WritingPrompts or ROCStories.
- **Why**: Coherence ensures generated text is understandable and contextually consistent, critical for user satisfaction.
- **Role**: Complements metrics like BLEU or ROUGE by focusing on the structural and logical quality of text.

**Advantages:**
- Captures high-level text quality (logical flow, narrative consistency).
- Can be automated with embeddings or language models.
- Aligns with human perception of text quality.

**Limitations:**
- No standardized computation, leading to variability.
- Automated methods may miss nuanced coherence issues.
- Human evaluation is subjective and resource-intensive.

**Example Use Case:**
In evaluating a dialogue model, Coherence Score uses sentence embedding similarity to ensure responses follow logically (e.g., “I’m tired” → “Let’s rest” is coherent).

---

### 9. Factual Accuracy

**Formula:**
No standard formula; often:
<img width="680" height="81" alt="image" src="https://github.com/user-attachments/assets/1a24edad-de38-47e0-9a30-2cd35b1e8444" />

- Factually correct: Verified against a knowledge base, reference text, or human judgment.
- Can be binary (correct/incorrect) or graded (e.g., partial correctness).

**Explanation:**
Factual Accuracy measures the proportion of generated statements that are factually correct, verified against ground truth sources (e.g., knowledge bases, Wikipedia, or human annotators). It ensures models avoid hallucinations or misinformation.

**When, Where, and Why It Is Used:**
- **When**: Used in tasks requiring factual correctness, such as QA, summarization, or knowledge-grounded dialogue.
- **Where**: Common in datasets like FEVER, TruthfulQA, or fact-checking benchmarks.
- **Why**: Ensures generated text aligns with verifiable facts, critical for applications like news generation or educational tools.
- **Role**: Assesses the reliability of generated content, complementing metrics like BLEU or BERTScore.

**Advantages:**
- Directly measures truthfulness, addressing a key ethical concern.
- Applicable to knowledge-intensive tasks.
- Can be verified using external sources or human evaluation.

**Limitations:**
- Requires reliable ground truth or fact-checking resources.
- Subjective in cases of ambiguous or context-dependent facts.
- Resource-intensive for large-scale evaluation.

**Example Use Case:**
In a news summarization model, Factual Accuracy verifies that a generated summary (e.g., “Joe Biden is the president in 2025”) matches verified facts, penalizing incorrect claims.

---

### Summary Table

| Metric                  | Focus                              | Task Type                      | Advantages                          | Limitations                         |
|-------------------------|------------------------------------|--------------------------------|-------------------------------------|-------------------------------------|
| F1 (QA/Task-specific)   | Precision-recall balance          | QA, NER, task-specific         | Balances precision/recall, flexible | Ignores semantics, matching-dependent |
| HumanEval (Pass@k)      | Code correctness                  | Code generation                | Measures functional correctness     | Test-case dependent, ignores quality |
| HellaSwag Accuracy      | Commonsense reasoning             | Commonsense reasoning          | Tests nuanced understanding         | Dataset-specific, multiple-choice   |
| MMLU                    | Multitask knowledge/reasoning     | General NLP evaluation         | Broad, standardized benchmark       | Limited to multiple-choice          |
| TruthfulQA Accuracy     | Truthfulness                      | Fact-based QA                  | Addresses misinformation            | Subjective truths, dataset-specific |
| Toxicity Score          | Safety/appropriateness            | Text generation, dialogue      | Quantifies ethical concerns         | Classifier-dependent, subjective    |
| Diversity (Distinct-n, Self-BLEU) | Text variety                   | Text generation                | Measures lexical diversity          | Misses semantic diversity           |
| Coherence Score         | Logical flow                     | Dialogue, story generation     | Captures high-level text quality    | Non-standardized, subjective        |
| Factual Accuracy        | Fact correctness                 | QA, summarization              | Ensures reliability                 | Resource-intensive, context-dependent |

---

### Detailed Understanding and Role in NLP
- **F1 (QA/Task-specific)** is critical for tasks requiring precise span or token identification, balancing completeness (recall) and correctness (precision). It’s widely used in QA and NER but limited by its focus on surface-level matches.
- **HumanEval (Pass@k)** is the gold standard for code generation, directly testing functional correctness. It’s practical for evaluating LLMs in programming but doesn’t assess code style or efficiency.
- **HellaSwag Accuracy** tests commonsense reasoning, pushing models to understand real-world scenarios. Its multiple-choice format limits its scope but makes it a challenging benchmark.
- **MMLU** is a comprehensive benchmark for general knowledge and reasoning, ideal for comparing LLMs across diverse domains, though it’s constrained by its multiple-choice structure.
- **TruthfulQA Accuracy** addresses the critical issue of misinformation, ensuring models provide reliable answers, especially in sensitive domains like health or law.
- **Toxicity Score** is essential for safe deployment, ensuring generated text is free of harmful content, though its reliance on classifiers introduces potential biases.
- **Diversity (Distinct-n, Self-BLEU)** quantifies variety in generated text, crucial for creative applications like story generation, but may miss semantic diversity.
- **Coherence Score** ensures logical flow, vital for user-facing applications like dialogue, though its computation varies and may require human judgment.
- **Factual Accuracy** ensures reliability in knowledge-intensive tasks, complementing other metrics by focusing on verifiable truth.

### When to Choose Each Metric
- Use **F1** for QA or task-specific evaluations where partial matches matter (e.g., SQuAD, NER).
- Use **HumanEval (Pass@k)** for code generation tasks to test functional correctness.
- Use **HellaSwag Accuracy** to evaluate commonsense reasoning in LLMs.
- Use **MMLU** for broad, multitask evaluation of general knowledge and reasoning.
- Use **TruthfulQA Accuracy** for tasks requiring factual reliability, especially with ambiguous questions.
- Use **Toxicity Score** for safety-critical applications like chatbots or content generation.
- Use **Diversity (Distinct-n, Self-BLEU)** for creative text generation to ensure varied outputs.
- Use **Coherence Score** for dialogue or narrative tasks where logical flow is key.
- Use **Factual Accuracy** for knowledge-intensive tasks like summarization or fact-based QA.

### Conclusion
These metrics cover a wide range of NLP evaluation needs, from task-specific accuracy (F1, HumanEval), commonsense reasoning (HellaSwag), general knowledge (MMLU), truthfulness (TruthfulQA), safety (Toxicity), diversity (Distinct-n, Self-BLEU), coherence, and factual correctness. Each metric addresses specific aspects of model performance, with F1 and HumanEval focusing on task-specific correctness, HellaSwag and MMLU testing reasoning and knowledge, and TruthfulQA, Toxicity, Diversity, Coherence, and Factual Accuracy ensuring ethical, varied, and reliable outputs. For comprehensive evaluation, multiple metrics are often combined to capture quality, functionality, and safety, depending on the task and application.
