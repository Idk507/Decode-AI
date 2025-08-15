
---

### 1. Perplexity

**Formula:**
<img width="952" height="169" alt="image" src="https://github.com/user-attachments/assets/15e54e11-70ea-4451-bf43-e263bd1cd2f1" />

**Explanation:**
Perplexity measures how well a probabilistic language model predicts a sequence of words. It is the exponentiation of the average negative log-likelihood of the sequence, reflecting the model’s uncertainty. A lower perplexity indicates better predictive performance, meaning the model assigns higher probabilities to the correct words.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate language models, particularly autoregressive models like GPT or n-gram models, during training or testing.
- **Where**: Common in language modeling, machine translation, and speech recognition tasks, especially on datasets like WikiText or Penn Treebank.
- **Why**: Perplexity quantifies how well the model captures the distribution of the target language, with lower values indicating better fit to the data.
- **Role**: Serves as a primary metric for assessing the quality of language models by measuring predictive accuracy. It’s especially useful for comparing models or tuning hyperparameters during training.

**Advantages:**
- Directly tied to the model’s probabilistic predictions, making it theoretically grounded.
- Easy to compute for probabilistic models.
- Widely used, enabling comparison across language models.

**Limitations:**
- Does not directly measure semantic quality, coherence, or fluency of generated text.
- Sensitive to vocabulary size and sequence length, which can skew comparisons.
- Less intuitive for non-technical audiences, as it’s not a direct measure of human-perceived quality.

**Example Use Case:**
In training a transformer-based language model on WikiText-103, perplexity is used to compare models, where a model with a perplexity of 20 is better at predicting the next word than one with a perplexity of 50.

---

### 2. BLEU (Bilingual Evaluation Understudy)

**Formula:**
<img width="990" height="466" alt="image" src="https://github.com/user-attachments/assets/316e92c1-0eb0-4f07-a7b8-a26fdd6162b0" />

**Explanation:**
BLEU measures the similarity between a machine-generated text (candidate) and one or more reference texts by computing the precision of n-grams (unigrams, bigrams, etc.) and applying a brevity penalty to penalize overly short outputs. It ranges from 0 to 1, with higher scores indicating better overlap.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate machine translation, text summarization, or other text generation tasks where word or phrase overlap is important.
- **Where**: Standard in machine translation (e.g., WMT datasets) and summarization tasks.
- **Why**: BLEU provides a quick, automated way to assess text quality by focusing on n-gram overlap, which correlates reasonably with human judgments for translation tasks.
- **Role**: Acts as a widely adopted benchmark for comparing translation or generation systems, emphasizing precision and length appropriateness.

**Advantages:**
- Simple to compute and widely used, enabling cross-system comparisons.
- Considers multiple n-grams, capturing both word choice and phrase structure.

**Limitations:**
- Focuses on exact n-gram matches, ignoring synonyms, paraphrases, or semantic meaning.
- Does not account for fluency or grammaticality.
- Sensitive to the number and quality of reference texts; multiple references improve reliability.

**Example Use Case:**
In evaluating a machine translation system from English to Spanish on the WMT dataset, BLEU compares the system’s output (e.g., “The cat is on the mat”) to human translations, rewarding exact n-gram matches.

---

### 3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE is a family of metrics (ROUGE-N, ROUGE-L, ROUGE-S, ROUGE-W) designed to evaluate text similarity, primarily in summarization and translation. Below, I focus on the most common variants: ROUGE-N and ROUGE-L.

#### ROUGE-N
**Formula:**
<img width="673" height="104" alt="image" src="https://github.com/user-attachments/assets/5ed01060-6b2e-4798-a193-f408e488e91b" />

<img width="807" height="121" alt="image" src="https://github.com/user-attachments/assets/53651922-6f9c-41b3-855f-f2b519bdc31d" />


**Explanation:**
ROUGE-N measures the recall of n-grams in the candidate text compared to the reference, focusing on how much of the reference content is captured.

#### ROUGE-L
**Formula:**
Based on the Longest Common Subsequence (LCS):
<img width="1045" height="329" alt="image" src="https://github.com/user-attachments/assets/65a1e6a2-0848-449b-a571-914a761f9658" />


**Explanation:**
ROUGE-L evaluates the longest common subsequence between candidate and reference, capturing sentence-level structure without requiring exact matches.

**When, Where, and Why It Is Used:**
- **When**: Used for text summarization, machine translation, or text generation tasks where recall of content is critical.
- **Where**: Common in summarization datasets (e.g., DUC, TAC) and machine translation evaluation.
- **Why**: ROUGE emphasizes recall, making it suitable for tasks where capturing the reference’s content is more important than exact wording. ROUGE-L is particularly useful for assessing structural similarity.
- **Role**: Standard metric for summarization, providing insights into content overlap (ROUGE-N) and sentence structure (ROUGE-L).

**Advantages:**
- ROUGE-N captures n-gram overlap; ROUGE-L captures sequence structure.
- Easy to compute and widely adopted in summarization research.
- Handles multiple references, improving robustness.

**Limitations:**
- Focuses on surface-level overlap, missing semantic similarity or fluency.
- Recall-oriented, potentially undervaluing precision.
- Dependent on reference quality and quantity.

**Example Use Case:**
In evaluating an abstractive summarization model on the CNN/DailyMail dataset, ROUGE-1, ROUGE-2, and ROUGE-L measure how well the generated summary captures key words, phrases, and sentence structure of the reference summary.

---

### 4. METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**Formula:**
<img width="623" height="89" alt="image" src="https://github.com/user-attachments/assets/5b3390ba-4594-4e2e-bc78-d3f96d62f8c8" />

- Precision: Fraction of matched unigrams in the candidate.
- Recall: Fraction of matched unigrams in the reference.
- Penalty: Penalizes fragmented alignments (chunkiness):
<img width="465" height="83" alt="image" src="https://github.com/user-attachments/assets/76db4de8-10b7-46cf-93d5-ce93ebb14b90" />

- Matches include exact words, stems, synonyms, and paraphrases (using resources like WordNet).

**Explanation:**
METEOR evaluates text similarity by matching unigrams (exact, stemmed, or synonymous) between the candidate and reference, and applies a penalty for fragmented alignments to account for word order and fluency. It balances precision and recall with a focus on semantic equivalence.

**When, Where, and Why It Is Used:**
- **When**: Used when semantic similarity and word order are important, particularly in machine translation.
- **Where**: Common in machine translation (e.g., WMT) and summarization tasks.
- **Why**: METEOR improves on BLEU by incorporating synonyms, stemming, and word order penalties, leading to better correlation with human judgments.
- **Role**: Provides a nuanced evaluation of translation or generation quality, capturing both semantic and structural aspects.

**Advantages:**
- Accounts for synonyms, stems, and paraphrases, improving semantic evaluation.
- Penalizes poor word order, rewarding fluency.
- Correlates better with human judgments than BLEU in many cases.

**Limitations:**
- Requires external linguistic resources (e.g., WordNet), increasing complexity.
- Computationally heavier than BLEU or ROUGE.
- Language-dependent due to reliance on resources like synonym databases.

**Example Use Case:**
In evaluating a machine translation system from English to French, METEOR rewards translations that use synonyms (e.g., “large” vs. “big”) and penalizes fragmented word alignments, providing a more human-aligned score than BLEU.

---

### 5. CIDEr (Consensus-based Image Description Evaluation)

**Formula:**
<img width="913" height="200" alt="image" src="https://github.com/user-attachments/assets/cd1d8059-4db6-4b4a-a17d-bcab2c6504cb" />
<img width="955" height="225" alt="image" src="https://github.com/user-attachments/assets/73a4015d-d53b-4e50-976f-925d09033922" />


**Explanation:**
CIDEr measures the similarity between a candidate text (e.g., image caption) and multiple reference texts, using TF-IDF weighting to emphasize informative n-grams and capture consensus among references. Higher CIDEr scores indicate better alignment with human-provided references.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate image captioning or text generation tasks with multiple valid references.
- **Where**: Standard in image captioning datasets like MS COCO.
- **Why**: CIDEr accounts for multiple references and downweights common n-grams (e.g., “the”), focusing on informative content, which aligns well with human judgments.
- **Role**: Primary metric for evaluating image captioning models, emphasizing consensus and informativeness.

**Advantages:**
- Handles multiple references, capturing diverse valid outputs.
- TF-IDF weighting emphasizes meaningful content.
- Correlates well with human judgments in image captioning.

**Limitations:**
- Computationally complex due to TF-IDF and multi-reference calculations.
- Assumes multiple references are available, which may not always be feasible.
- Focused on n-gram overlap, missing deeper semantic nuances.

**Example Use Case:**
In evaluating an image captioning model on MS COCO, CIDEr assesses how well a generated caption (e.g., “A dog plays in the park”) matches multiple human captions for the same image.

---

### 6. BERTScore

**Formula:**
<img width="1027" height="302" alt="image" src="https://github.com/user-attachments/assets/0f7aad61-b18a-4194-b395-d60bbf1caef3" />

**Explanation:**
BERTScore uses contextual embeddings from a pre-trained model (e.g., BERT) to compute token-level similarity between the candidate and reference texts. It calculates precision, recall, and F1-score based on the maximum cosine similarity of token embeddings, capturing semantic similarity.

**When, Where, and Why It Is Used:**
- **When**: Used when semantic similarity is more important than surface-level overlap.
- **Where**: Common in machine translation, summarization, and text generation tasks (e.g., on WMT or CNN/DailyMail datasets).
- **Why**: BERTScore leverages contextual embeddings to capture meaning, making it robust to paraphrases and synonyms, unlike BLEU or ROUGE.
- **Role**: Evaluates semantic quality of generated text, providing a more human-aligned measure than n-gram-based metrics.

**Advantages:**
- Captures semantic similarity, robust to paraphrases and synonyms.
- Correlates well with human judgments.
- Flexible across tasks and languages (with multilingual BERT models).

**Limitations:**
- Computationally expensive due to embedding generation.
- Dependent on the quality of the pre-trained model (e.g., BERT).
- Less interpretable than n-gram metrics like BLEU or ROUGE.

**Example Use Case:**
In evaluating a paraphrasing model, BERTScore assesses whether the generated paraphrase (e.g., “The house is big”) conveys the same meaning as the reference (e.g., “The residence is large”), even with different wording.

---

### 7. MoverScore

**Formula:**
<img width="544" height="58" alt="image" src="https://github.com/user-attachments/assets/4478c03e-7dec-4324-8601-2cbab2a07d74" />

- $\( \text{WMD} \)$ : Word Mover’s Distance, the Earth Mover’s Distance between word embeddings of the candidate and reference, measuring the cost of transforming one text into another:
<img width="412" height="89" alt="image" src="https://github.com/user-attachments/assets/e9f7c66e-29de-424b-88dc-7c5e2b8ecc56" />

- $\( \mathbf{T} \)$ : Transport matrix, $\( \text{cost}(x_i, y_j) \)$ : Distance between embeddings (e.g., cosine distance).
- Embeddings are typically from a pre-trained model like BERT.

**Explanation:**
MoverScore extends BERTScore by using Word Mover’s Distance to compute the semantic distance between candidate and reference texts, based on contextual embeddings. It captures global semantic alignment, considering the entire text rather than token-level matches.

**When, Where, and Why It Is Used:**
- **When**: Used for evaluating semantic similarity in tasks requiring robust meaning preservation.
- **Where**: Common in machine translation, summarization, and dialogue systems.
- **Why**: MoverScore accounts for global semantic relationships, making it more robust than token-level metrics like BERTScore for certain tasks.
- **Role**: Provides a sophisticated measure of semantic similarity, particularly for tasks where word order or global context matters.

**Advantages:**
- Captures global semantic alignment, not just token-level similarity.
- Robust to paraphrases, synonyms, and word order variations.
- Correlates well with human judgments.

**Limitations:**
- Computationally intensive due to WMD calculation.
- Requires high-quality embeddings from a pre-trained model.
- Less widely adopted than BERTScore, limiting comparability.

**Example Use Case:**
In evaluating a dialogue system, MoverScore assesses whether a generated response (e.g., “I’m thrilled to join!”) conveys the same meaning as a reference response (e.g., “I’m excited to participate!”), even with different phrasing.

---

### 8. Exact Match (EM)

**Formula:**
<img width="413" height="106" alt="image" src="https://github.com/user-attachments/assets/5257eae9-e77e-4a3c-9099-bf507536a16f" />

- Exact match: 1 if the candidate text exactly matches the reference text (after normalization, e.g., lowercasing, punctuation removal), 0 otherwise.

**Explanation:**
EM measures the proportion of predictions that exactly match the reference text or answer, typically used in tasks requiring verbatim correctness, such as question answering.

**When, Where, and Why It Is Used:**
- **When**: Used when exact correctness is critical, and partial matches are not sufficient.
- **Where**: Common in question answering (e.g., SQuAD, TriviaQA) and structured prediction tasks.
- **Why**: EM is a strict metric that ensures the model produces the exact expected output, useful for tasks with a single correct answer.
- **Role**: Evaluates whether a model produces verbatim correct answers, often alongside more lenient metrics like F1.

**Advantages:**
- Simple and strict, ensuring perfect matches.
- Easy to compute and interpret as a percentage.

**Limitations:**
- No credit for partial correctness or semantic similarity (e.g., synonyms or paraphrases).
- Harsh for tasks where multiple valid answers exist.
- Sensitive to minor differences (e.g., punctuation, case).

**Example Use Case:**
In evaluating a question answering model on SQuAD (e.g., question: “Who is the president of the USA?”), EM checks if the model’s answer exactly matches “Joe Biden” (as of August 15, 2025).

---

### Summary Table

| Metric       | Focus                              | Task Type                      | Advantages                          | Limitations                         |
|--------------|------------------------------------|--------------------------------|-------------------------------------|-------------------------------------|
| Perplexity   | Language model uncertainty        | Language modeling              | Probabilistic, easy to compute      | Ignores semantics, not intuitive    |
| BLEU         | N-gram precision                 | Machine translation, summarization | Simple, widely used                 | Ignores synonyms, sensitive to references |
| ROUGE (N/L)  | N-gram/LCS recall                | Summarization, translation     | Captures content/structure          | Surface-level, recall-oriented      |
| METEOR       | Semantic unigram matching        | Translation, summarization     | Accounts for synonyms, fluency      | Resource-heavy, language-dependent  |
| CIDEr        | Consensus-based n-gram similarity | Image captioning              | Emphasizes informative content      | Requires multiple references        |
| BERTScore    | Semantic similarity (embeddings) | Translation, summarization     | Captures semantics, robust to paraphrases | Computationally heavy, model-dependent |
| MoverScore   | Global semantic alignment        | Translation, summarization     | Robust semantic evaluation          | Computationally intensive           |
| Exact Match  | Verbatim correctness             | Question answering            | Strict, simple                      | Harsh, no partial credit           |

---

### Detailed Understanding and Role in NLP
- **Perplexity** is foundational for language model training, guiding optimization by measuring how well the model predicts text. It’s less relevant for evaluating generated text quality in human terms but critical for probabilistic models.
- **BLEU** is a cornerstone for machine translation due to its simplicity and focus on n-gram precision, but its reliance on exact matches limits its ability to handle semantic variations, making it less ideal for tasks like summarization.
- **ROUGE** is the go-to metric for summarization, emphasizing content recall. ROUGE-N is effective for word/phrase overlap, while ROUGE-L captures structural similarity, making it versatile for various text generation tasks.
- **METEOR** improves on BLEU by incorporating semantic matching (synonyms, stems) and word order, offering a more human-aligned evaluation for translation and summarization.
- **CIDEr** is tailored for image captioning, leveraging multiple references and TF-IDF to prioritize informative content, making it highly effective for tasks with diverse valid outputs.
- **BERTScore** and **MoverScore** represent modern, embedding-based metrics that prioritize semantic similarity. BERTScore is token-focused and widely applicable, while MoverScore’s global alignment makes it suited for tasks requiring coherent text evaluation.
- **Exact Match** is a strict metric for question answering, ensuring exact correctness but lacking flexibility for partial matches, often used alongside metrics like F1 for a balanced evaluation.

### When to Choose Each Metric
- Use **Perplexity** for training and evaluating language models, especially when optimizing for predictive accuracy.
- Use **BLEU** for machine translation tasks where exact n-gram overlap is a reasonable proxy for quality.
- Use **ROUGE** for summarization or tasks where content recall is critical.
- Use **METEOR** for translation or summarization when semantic flexibility and word order matter.
- Use **CIDEr** for image captioning to capture consensus among multiple references.
- Use **BERTScore** or **MoverScore** for tasks requiring semantic evaluation, such as paraphrasing or dialogue, with MoverScore preferred for global coherence.
- Use **EM** for question answering or tasks where exact answers are required.

### Conclusion
These metrics collectively cover a range of NLP evaluation needs, from probabilistic modeling (Perplexity) to surface-level overlap (BLEU, ROUGE), semantic similarity (METEOR, BERTScore, MoverScore), consensus-based captioning (CIDEr), and strict correctness (EM). The choice of metric depends on the task, data, and evaluation goals, with modern metrics like BERTScore and MoverScore offering superior semantic alignment, while traditional metrics like BLEU and ROUGE remain valuable for their simplicity and standardization. For a comprehensive evaluation, multiple metrics are often used together to capture different aspects of performance.
