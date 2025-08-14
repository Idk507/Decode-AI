
---

### 1. BLEU (Bilingual Evaluation Understudy)

**Formula:**
<img width="429" height="111" alt="image" src="https://github.com/user-attachments/assets/97e9964d-ce95-49f1-9f7c-7f70ae772ae4" />

where:
- $\( p_n \)$ : Precision for n-grams (ratio of matching n-grams in the candidate to the total n-grams in the candidate).
- $\( w_n \)$ : Weight for each n-gram order (typically $\( w_n = 1/N \), e.g., \( 1/4 \)$ for n=1 to 4).
- $\( \text{BP} \)$ : Brevity penalty, to penalize short translations:
<img width="420" height="101" alt="image" src="https://github.com/user-attachments/assets/35d905a4-8390-4a61-8b88-4d33a953e46f" />

- $\( c \)$ : Length of the candidate translation.
- $\( r \)$ : Length of the reference translation.

**Explanation:**
BLEU measures the similarity between a machine-generated text (candidate) and one or more reference texts by computing the precision of n-grams (unigrams, bigrams, etc.) and applying a brevity penalty to discourage overly short outputs.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate machine translation and text generation tasks where exact word or phrase overlap with references is important.
- **Where**: Common in machine translation (e.g., Google Translate evaluation) and summarization.
- **Why**: BLEU is a simple, widely adopted metric that correlates reasonably well with human judgments of translation quality for n-gram overlap.
- **Role**: Provides a quick, automated way to assess the quality of generated text by focusing on precision and length.

**Advantages:**
- Easy to compute and widely used, enabling comparison across systems.
- Considers multiple n-grams, capturing both word choice and phrase structure.

**Limitations:**
- Focuses on exact n-gram matches, ignoring synonyms or paraphrases.
- Does not account for fluency, grammaticality, or semantic correctness.
- Sensitive to the choice of reference texts; multiple references improve reliability.

**Example Use Case:**
Evaluating a machine translation system translating English to French, where BLEU compares the system’s output to human-translated references.

---

### 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE is a family of metrics (ROUGE-N, ROUGE-L, ROUGE-S, ROUGE-W) used to evaluate text similarity, primarily in summarization and machine translation.

#### ROUGE-N
**Formula:**
<img width="632" height="89" alt="image" src="https://github.com/user-attachments/assets/8f6af705-c3a1-4922-83de-6d60e2df1ea8" />

- $\( \text{gram}_n \)$: N-grams of order $\( n \)$.
- $\( \text{Count}_{\text{match}}(\text{gram}_n) \)$ : Number of n-grams in both candidate and reference.
- Denominator: Total n-grams in the reference.

**Explanation:**
ROUGE-N measures the recall of n-grams (e.g., ROUGE-1 for unigrams, ROUGE-2 for bigrams) in the candidate text compared to the reference.

#### ROUGE-L
**Formula:**
Based on the Longest Common Subsequence (LCS):
<img width="972" height="315" alt="image" src="https://github.com/user-attachments/assets/ad78ec8d-5b86-487c-a4fe-d277cb2743d1" />


**Explanation:**
ROUGE-L evaluates the longest common subsequence between candidate and reference, capturing sentence-level structure without requiring exact matches.

#### ROUGE-S
**Formula:**
Based on skip-bigrams (bigrams allowing gaps):
<img width="743" height="126" alt="image" src="https://github.com/user-attachments/assets/be9db7aa-3d27-43a7-a95d-d7d9f968a479" />


**Explanation:**
ROUGE-S measures recall of skip-bigrams (pairs of words in order, with gaps allowed), capturing word order flexibility.

#### ROUGE-W
**Explanation:**
ROUGE-W extends ROUGE-L by weighting consecutive matches in the LCS more heavily, using a weighted LCS to emphasize fluency and coherence.

**When, Where, and Why It Is Used:**
- **When**: Used for evaluating text summarization, machine translation, or text generation tasks where recall is important.
- **Where**: Common in summarization (e.g., DUC, TAC datasets), machine translation, and question answering.
- **Why**: ROUGE focuses on recall, making it suitable for tasks where capturing reference content is critical. Variants like ROUGE-L and ROUGE-S handle structural and flexible matches.
- **Role**: ROUGE is a standard metric for summarization, providing insights into content overlap and structure.

**Advantages:**
- ROUGE-N captures n-gram overlap, ROUGE-L captures sentence structure, and ROUGE-S allows flexibility in word order.
- Easy to compute and widely adopted.

**Limitations:**
- Focuses on surface-level overlap, missing semantic similarity.
- Recall-oriented, so may undervalue precision or fluency.
- Dependent on reference quality and quantity.

**Example Use Case:**
In evaluating an abstractive summarization model, ROUGE-1, ROUGE-2, and ROUGE-L assess how well the summary captures key unigrams, bigrams, and sentence structure of the reference.

---

### 3. METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**Formula:**
<img width="584" height="98" alt="image" src="https://github.com/user-attachments/assets/3465634b-30d6-4d7f-a4ff-bde28fbb8e8b" />

- Precision: Fraction of matched unigrams in the candidate.
- Recall: Fraction of matched unigrams in the reference.
- Penalty: Penalizes chunkiness (fragmentation) of matches:
<img width="456" height="73" alt="image" src="https://github.com/user-attachments/assets/34e4719d-2796-4a32-86d8-dc748fb32989" />

- Matches include exact words, stems, synonyms, and paraphrases (using external resources like WordNet).

**Explanation:**
METEOR evaluates text similarity by matching unigrams (exact, stemmed, or synonymous) and penalizing fragmented alignments to account for word order and fluency.

**When, Where, and Why It Is Used:**
- **When**: Used when semantic similarity and word order are important.
- **Where**: Common in machine translation and summarization, especially when BLEU’s exact-match focus is insufficient.
- **Why**: METEOR incorporates synonyms and stemming, making it more flexible than BLEU, and penalizes poor word order, improving correlation with human judgments.
- **Role**: Provides a more nuanced evaluation of translation quality by considering semantics and structure.

**Advantages:**
- Accounts for synonyms, stems, and paraphrases, capturing semantic similarity.
- Penalizes fragmentation, rewarding fluency.
- Correlates better with human judgments than BLEU in many cases.

**Limitations:**
- Requires external resources (e.g., WordNet), increasing complexity.
- Computationally heavier than BLEU or ROUGE.
- Language-dependent due to reliance on linguistic resources.

**Example Use Case:**
In machine translation evaluation, METEOR assesses a system’s output by matching synonyms (e.g., “big” and “large”) and penalizing poorly ordered translations.

---

### 4. CIDEr (Consensus-based Image Description Evaluation)

**Formula:**
<img width="906" height="192" alt="image" src="https://github.com/user-attachments/assets/b0a779e8-d38c-4e13-b6e2-3c32699af194" />

- $\( \text{Count}_i(\text{gram}_n) \)$: Count of n-grams in the candidate.
- $\( \text{Count}_{\text{ref}}(\text{gram}_n) \)$: Count of n-grams in the reference.
- $\( \text{TF-IDF}(\text{gram}_n) \)$ : Term frequency-inverse document frequency weight for n-grams.
- $\( m \)$ : Number of reference descriptions.
- $\( w_n \)$ : Weight for n-gram order (e.g., uniform for n=1 to 4).

**Explanation:**
CIDEr measures the similarity of a candidate text (e.g., image caption) to multiple reference texts, using TF-IDF weighting to emphasize informative n-grams and account for consensus among references.

**When, Where, and Why It Is Used:**
- **When**: Used for evaluating image captioning or text generation tasks requiring consensus among multiple references.
- **Where**: Common in image captioning (e.g., MS COCO dataset) and other tasks with multiple valid references.
- **Why**: CIDEr uses TF-IDF to downweight common n-grams (e.g., “the”) and focus on informative content, improving alignment with human judgments.
- **Role**: Evaluates how well a generated text captures the consensus of multiple reference descriptions.

**Advantages:**
- Accounts for multiple references, capturing diverse valid outputs.
- TF-IDF weighting emphasizes informative content.
- Correlates well with human judgments in image captioning.

**Limitations:**
- Complex to compute due to TF-IDF and multiple n-gram orders.
- Assumes multiple references are available, which may not always be the case.

**Example Use Case:**
In image captioning, CIDEr evaluates how well a generated caption (e.g., “A dog runs on grass”) matches multiple human-provided captions for the same image.

---

### 5. Perplexity

**Formula:**
<img width="693" height="140" alt="image" src="https://github.com/user-attachments/assets/60d1f2b8-257c-4905-946d-03bf95b538a1" />


**Explanation:**
Perplexity measures how well a language model predicts a sequence of words, with lower perplexity indicating better predictive performance. It is the exponentiation of the average negative log-likelihood of the sequence.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate language models (e.g., GPT, BERT) on their ability to predict text.
- **Where**: Common in language modeling, machine translation, and speech recognition.
- **Why**: Perplexity quantifies the uncertainty of a model’s predictions, with lower values indicating better modeling of the data distribution.
- **Role**: Used to assess the quality of probabilistic language models during training or evaluation.

**Advantages:**
- Directly tied to the model’s probabilistic predictions.
- Widely used for comparing language models.

**Limitations:**
- Does not directly measure semantic quality or coherence.
- Sensitive to vocabulary size and text length.
- Not intuitive for non-technical audiences.

**Example Use Case:**
In training a language model for text generation, perplexity is used to compare models, where a lower perplexity indicates better word prediction.

---

### 6. Word Error Rate (WER)

**Formula:**
<img width="270" height="81" alt="image" src="https://github.com/user-attachments/assets/ed4713bc-21b6-4036-8a7c-995c7808765f" />

- $\( S \)$: Number of substitutions (wrong words).
- $\( D \)$: Number of deletions (missing words).
- $\( I \)$: Number of insertions (extra words).
- $\( N \)$ : Number of words in the reference.

**Explanation:**
WER measures the error rate in a transcription or translation by counting the minimum number of edit operations (substitutions, deletions, insertions) needed to transform the candidate into the reference, normalized by the reference length.

**When, Where, and Why It Is Used:**
- **When**: Used in speech recognition, machine translation, or text generation to evaluate word-level accuracy.
- **Where**: Common in automatic speech recognition (ASR) and machine translation systems.
- **Why**: WER provides a direct measure of word-level errors, making it useful for tasks where exact word matches are critical.
- **Role**: Quantifies transcription or translation accuracy at the word level.

**Advantages:**
- Simple and interpretable as a percentage of errors.
- Captures multiple types of errors (substitutions, deletions, insertions).

**Limitations:**
- Does not account for semantic similarity (e.g., synonyms).
- Sensitive to word order and minor variations.
- Harsh on small differences that may not affect meaning.

**Example Use Case:**
In evaluating a speech-to-text system, WER measures errors in the transcribed text compared to the ground truth (e.g., “I run” vs. “I ran”).

---

### 7. Character Error Rate (CER)

**Formula:**
<img width="332" height="71" alt="image" src="https://github.com/user-attachments/assets/d0046945-60e7-4f0f-8e08-f71af292b7c3" />

- $\( S_c \)$: Number of character substitutions.
- $\( D_c \)$ : Number of character deletions.
- $\( I_c \)$ : Number of character insertions.
- $\( N_c \)$ : Number of characters in the reference.

**Explanation:**
CER is similar to WER but operates at the character level, measuring the minimum number of edit operations needed to align the candidate text with the reference.

**When, Where, and Why It Is Used:**
- **When**: Used in tasks where character-level accuracy is important, such as speech recognition or OCR.
- **Where**: Common in automatic speech recognition, optical character recognition (OCR), and spelling correction.
- **Why**: CER is more granular than WER, making it suitable for languages with complex scripts or when small character differences matter.
- **Role**: Evaluates fine-grained accuracy in text output.

**Advantages:**
- More granular than WER, capturing small differences.
- Useful for languages with non-word-based scripts (e.g., Chinese).

**Limitations:**
- Does not capture semantic or word-level errors.
- Sensitive to minor typos that may not affect meaning.

**Example Use Case:**
In OCR for handwritten text, CER evaluates how accurately characters are recognized compared to the reference text.

---

### 8. BERTScore

**Formula:**
<img width="974" height="179" alt="image" src="https://github.com/user-attachments/assets/e97dfb78-7c47-4199-8927-5610718783d1" />

- $\( \mathbf{x}_i \)$ : Contextual embedding of token $\( i \)$ in the candidate.
- $\( \mathbf{y}_j \)$ : Contextual embedding of token $\( j \)$ in the reference.
- $\( \cos \)$ : Cosine similarity between embeddings.

**Explanation:**
BERTScore uses contextual embeddings (e.g., from BERT) to compute the similarity between tokens in the candidate and reference texts, focusing on semantic similarity rather than exact matches.

**When, Where, and Why It Is Used:**
- **When**: Used when semantic similarity is more important than surface-level overlap.
- **Where**: Common in machine translation, summarization, and text generation tasks.
- **Why**: BERTScore captures contextual and semantic similarity, making it more robust to paraphrases and synonyms than BLEU or ROUGE.
- **Role**: Evaluates semantic quality of generated text using deep learning embeddings.

**Advantages:**
- Captures semantic similarity via contextual embeddings.
- Robust to paraphrases and synonyms.
- Correlates well with human judgments.

**Limitations:**
- Computationally expensive due to embedding generation.
- Dependent on the quality of the embedding model (e.g., BERT).
- Less interpretable than n-gram-based metrics.

**Example Use Case:**
In evaluating a paraphrasing model, BERTScore assesses whether the generated paraphrase retains the meaning of the reference, even if worded differently.

---

### 9. MoverScore

**Formula:**
<img width="576" height="61" alt="image" src="https://github.com/user-attachments/assets/f93c48b8-3949-4bcb-985d-13ca7bf781b7" />

- WMD (Word Mover’s Distance): Earth Mover’s Distance between word embeddings of the candidate and reference, measuring the cost of transforming one text into another.

**Explanation:**
MoverScore extends BERTScore by using Word Mover’s Distance to compute the semantic distance between candidate and reference texts, based on contextual embeddings.

**When, Where, and Why It Is Used:**
- **When**: Used for evaluating semantic similarity in tasks requiring robust meaning preservation.
- **Where**: Common in machine translation, summarization, and dialogue systems.
- **Why**: MoverScore considers the global alignment of embeddings, capturing semantic relationships more effectively than token-level metrics.
- **Role**: Provides a sophisticated measure of semantic similarity.

**Advantages:**
- Captures global semantic alignment, not just token-level similarity.
- Robust to paraphrases and word order variations.

**Limitations:**
- Computationally intensive due to WMD calculation.
- Requires high-quality embeddings.
- Less widely adopted than BERTScore.

**Example Use Case:**
In dialogue systems, MoverScore evaluates whether a generated response conveys the same meaning as a reference response, even with different phrasing.

---

### 10. TER (Translation Edit Rate)

**Formula:**
<img width="347" height="97" alt="image" src="https://github.com/user-attachments/assets/e2bc00bd-80de-4e08-ac01-72b2b273d16d" />

- $\( S \)$ : Substitutions.
- $\( D \)$ : Deletions.
- $\( I \)$ : Insertions.
- $\( R \)$ : Shifts (reordering of phrases).
- $\( N \)$ : Number of words in the reference.

**Explanation:**
TER extends WER by including shifts (reordering of word sequences), measuring the minimum number of edits needed to align the candidate with the reference.

**When, Where, and Why It Is Used:**
- **When**: Used in machine translation to evaluate translation quality, accounting for word order changes.
- **Where**: Common in translation evaluation, especially in post-editing scenarios.
- **Why**: TER captures reordering, which is common in translation, making it more flexible than WER.
- **Role**: Quantifies the effort needed to correct a translation.

**Advantages:**
- Accounts for phrase reordering, unlike WER.
- Useful for post-editing analysis in translation workflows.

**Limitations:**
- Computationally complex due to shift calculations.
- Does not capture semantic similarity.
- Sensitive to reference text choice.

**Example Use Case:**
In evaluating a translation system, TER measures the edits (including reordering) needed to match the reference translation.

---

### 11. F1 (Token/Entity-level for NER)

**Formula:**
<img width="598" height="168" alt="image" src="https://github.com/user-attachments/assets/50a967cc-e055-4bbc-a1dd-3a28c62ccf58" />

- TP: True Positives (correctly identified entities/tokens).
- FP: False Positives (incorrectly identified entities/tokens).
- FN: False Negatives (missed entities/tokens).

**Explanation:**
F1 for Named Entity Recognition (NER) measures the balance between precision and recall for identifying entities (e.g., person, organization) at the token or entity level. Token-level F1 evaluates individual token labels, while entity-level F1 evaluates entire entity spans.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate NER or sequence labeling tasks.
- **Where**: Common in NER, part-of-speech tagging, and chunking.
- **Why**: F1 balances precision (avoiding false positives) and recall (capturing all true entities), providing a single metric for model performance.
- **Role**: Standard metric for evaluating entity extraction accuracy.

**Advantages:**
- Balances precision and recall, avoiding bias toward either.
- Applicable at both token and entity levels.

**Limitations:**
- Does not account for partial matches (e.g., partially correct entity boundaries).
- May require strict or relaxed matching criteria, affecting results.

**Example Use Case:**
In NER for identifying person names in text, F1 measures how accurately the model identifies “Barack Obama” as a single entity.

---

### 12. Exact Match (EM)

**Formula:**
<img width="402" height="100" alt="image" src="https://github.com/user-attachments/assets/9f292b73-6845-4894-ae33-3a6b792c7bb0" />

- Exact match: 1 if the candidate exactly matches the reference, 0 otherwise.

**Explanation:**
EM measures the proportion of predictions that exactly match the reference text or answer, often used in question answering or structured prediction tasks.

**When, Where, and Why It Is Used:**
- **When**: Used when exact correctness is critical.
- **Where**: Common in question answering (e.g., SQuAD dataset) and structured prediction.
- **Why**: EM is a strict metric that ensures the model produces the exact expected output.
- **Role**: Evaluates whether a model produces verbatim correct answers.

**Advantages:**
- Simple and strict, ensuring perfect matches.
- Easy to interpret as a percentage.

**Limitations:**
- No credit for partial correctness or semantic similarity.
- Harsh for tasks where multiple valid answers exist.

**Example Use Case:**
In question answering (e.g., “Who is the president?”), EM checks if the model’s answer exactly matches “Joe Biden.”

---

### 13. Spearman Correlation

**Formula:**
<img width="283" height="97" alt="image" src="https://github.com/user-attachments/assets/4e2250f5-1d4f-41fe-a1e5-c6bc076affd4" />

- $\( d_i \)$ : Difference between the ranks of corresponding values in two variables.
- $\( n \)$ : Number of observations.

**Explanation:**
Spearman Correlation measures the monotonic relationship between two ranked variables (e.g., model predictions and human judgments), assessing how well the model’s ranking aligns with the ground truth.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate ranking or ordinal predictions.
- **Where**: Common in tasks like text similarity, sentiment analysis, or evaluating metric-human judgment alignment.
- **Why**: Spearman is non-parametric, robust to non-linear relationships, and focuses on ranking rather than absolute values.
- **Role**: Assesses the alignment of model predictions with human evaluations or rankings.

**Advantages:**
- Robust to non-linear relationships and outliers.
- Suitable for ordinal data.

**Limitations:**
- Does not measure absolute agreement, only rank correlation.
- Less informative for non-ranking tasks.

**Example Use Case:**
In evaluating a text similarity model, Spearman Correlation compares the model’s similarity scores to human-assigned similarity rankings.

---

### 14. GLUE Score (General Language Understanding Evaluation)

**Explanation:**
GLUE Score is an aggregate metric averaging performance across multiple tasks in the GLUE benchmark (e.g., SST-2, MNLI, QQP). Each task uses its own metric (e.g., accuracy, F1, correlation), and the GLUE Score is the unweighted average of these scores.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate general language understanding across diverse NLP tasks.
- **Where**: Common in benchmarking language models (e.g., BERT, RoBERTa) on the GLUE dataset.
- **Why**: GLUE Score provides a single metric to compare model performance across varied tasks, testing robustness and generalization.
- **Role**: Standard benchmark for evaluating pre-trained language models.

**Advantages:**
- Comprehensive, covering multiple NLP tasks.
- Widely used, enabling comparison across models.

**Limitations:**
- Aggregate nature masks task-specific performance.
- Limited to the tasks in the GLUE benchmark, which may not cover all NLP scenarios.

**Example Use Case:**
In comparing BERT and RoBERTa, the GLUE Score summarizes their performance across tasks like sentiment analysis and natural language inference.

---

### Summary Table

| Metric            | Focus                              | Task Type                      | Advantages                          | Limitations                         |
|-------------------|------------------------------------|--------------------------------|-------------------------------------|-------------------------------------|
| BLEU              | N-gram precision                  | Machine translation, summarization | Simple, widely used                 | Ignores semantics, sensitive to references |
| ROUGE (N/L/S/W)   | N-gram/LCS/skip-bigram recall     | Summarization, translation     | Captures content/structure          | Surface-level, recall-oriented      |
| METEOR            | Semantic unigram matching         | Translation, summarization     | Accounts for synonyms, fluency      | Resource-heavy, language-dependent  |
| CIDEr             | Consensus-based n-gram similarity | Image captioning               | Emphasizes informative content      | Requires multiple references        |
| Perplexity        | Language model uncertainty        | Language modeling              | Tied to probabilistic predictions   | Ignores semantics, not intuitive    |
| WER               | Word-level edit errors            | Speech recognition, translation | Simple, captures word errors        | Ignores semantics, sensitive to order |
| CER               | Character-level edit errors       | Speech recognition, OCR        | Granular, good for complex scripts   | Ignores semantics, sensitive to typos |
| BERTScore         | Semantic similarity (embeddings)  | Translation, summarization     | Captures semantics, robust to paraphrases | Computationally heavy, model-dependent |
| MoverScore        | Global semantic alignment         | Translation, summarization     | Robust semantic evaluation          | Computationally intensive           |
| TER               | Edit errors with shifts           | Machine translation            | Accounts for reordering             | Complex, ignores semantics          |
| F1 (NER)          | Precision-recall balance          | NER, sequence labeling         | Balances precision and recall       | No partial credit for boundaries    |
| Exact Match (EM)  | Verbatim correctness              | Question answering             | Strict, simple                      | Harsh, no partial credit           |
| Spearman Corr.    | Ranking alignment                 | Similarity, ranking tasks      | Robust to non-linear relationships  | Rank-focused, not absolute agreement |
| GLUE Score        | Aggregate task performance        | General NLP evaluation         | Comprehensive benchmark             | Masks task-specific performance     |

---

### Conclusion
Each NLP evaluation metric serves a specific purpose depending on the task and evaluation needs. BLEU, ROUGE, and METEOR are suited for surface-level or semantic overlap in translation and summarization, while CIDEr excels in image captioning. Perplexity evaluates language model quality, and WER/CER are ideal for transcription tasks. BERTScore and MoverScore leverage embeddings for semantic evaluation, while TER and F1 focus on edit-based and entity-level accuracy. EM and Spearman Correlation address strict correctness and ranking tasks, respectively, and GLUE Score provides a broad benchmark for general NLP performance. Choosing the right metric depends on the task, data, and whether semantic, structural, or exact matches are prioritized.
