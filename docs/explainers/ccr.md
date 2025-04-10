# Chance-confidence ratio

Logion's [error detection](../user-guide/detection.md) algorithm uses what's called the chance-confidence ratio. Logion team members developed and introduced the algorithm in a 2023 [ACL paper](https://aclanthology.org/2023.alp-1.20/). The algorithm determines the likelihood that a given word has been mistranscribed at a certain point in the textual history.

## The algorithm

### Chance

Chance is the probability that an actual word in an existing text appears in a sentence given all of the other words in that sentence.

Chance is represented by the equation $p(w_i|w_{-i})$. Here, $p$ represents *probability*. $w_i$ represents one given word in a text. $w_{-i}$ represents that word's context—i.e. all of the other words in that sentence except for $w$. The algorithm calculates a word's chance by predicting the probability of that word given all of the surrounding words in the same sentence.

### Confidence

Confidence is the probability that a word suggested by a language model appears in a sentence given all of the pre-existing words in that sentence.

Confidence is represented by the equation $\underset{\text{word } w}{\max} p(w|w_{-i})$. Here, $\underset{\text{word } w}{\max}$ essentially represents the highest-ranked word suggested by a language model. $w$ represents that highest-ranked word. This word is inserted into the original sentence, replacing the original word $w_i$. The algorithm then determines the probability of that new word $w$ given all of the other words in the sentence (represented by $w_{-i}$).

Logion filters model suggestions for each word using [Levenshtein distance](lev-dist.md).

### The ratio

The algorithm then essentially divides the original word's chance score by the confidence score of its highest-suggested replacement. The result is the chance-confidence ratio for that word pair.
