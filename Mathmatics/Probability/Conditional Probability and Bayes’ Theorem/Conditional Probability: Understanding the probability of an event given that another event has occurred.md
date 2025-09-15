Conditional Probability Summary
Definition

Conditional Probability: The probability of event ( A ) occurring given that event ( B ) has occurred, denoted ( P(A | B) ).
Formula:[P(A | B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0]
Intuition: Restricts the sample space to ( B ), computing the fraction of ( B )’s outcomes that satisfy ( A ).

Key Properties

Range: ( 0 \leq P(A | B) \leq 1 ).
Normalization: ( P(A | \Omega) = P(A) ).
Multiplication Rule: ( P(A \cap B) = P(A | B) \cdot P(B) ).
Independence: If ( A ) and ( B ) are independent, ( P(A | B) = P(A) ), and ( P(A \cap B) = P(A) \cdot P(B) ).

Examples

Die Roll:
( A ): Roll an even number (( {2, 4, 6} )).
( B ): Roll > 3 (( {4, 5, 6} )).
( P(A | B) = \frac{P({4, 6})}{P({4, 5, 6})} = \frac{2/6}{3/6} = \frac{2}{3} ).


Cards:
( A ): Draw a heart.
( B ): Draw a red card.
( P(A | B) = \frac{P(\text{heart})}{P(\text{red})} = \frac{13/52}{26/52} = \frac{1}{2} ).


Medical Test:
( P(D) = 0.01 ), ( P(T^+ | D) = 0.95 ), ( P(T^+ | D^c) = 0.05 ).
( P(D | T^+) \approx 0.161 ) (using Bayes’ Theorem).



Applications

Updating Probabilities: Adjusts likelihoods based on new information.
Bayes’ Theorem: Updates beliefs (e.g., medical diagnostics, machine learning).
Independence Testing: Checks if events are related.
Fields: Medicine, finance, data science, decision-making.
