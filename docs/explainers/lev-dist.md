# Levenshtein distance

Levenshtein distance is a term from natural language processing (NLP). It refers to the minimum number of letter-level alterations needed to transform one word into another.

For example, say we transform *δε* to *τε*. This transformation requires changing one letter in the original word (*δ* **->** *τ*) to obtain the new word. Because the transformation requires changing only one letter, *δε* and *τε* are 1 Levenshtein distance apart.

## How Logion uses Levenshtein distance

The Logion app uses Levenshtein distance to filter suggestions for [error detection](../how-to/detection.md). During error detection, a model looks for misspelled or mistranscribed words and suggests potential replacements. Our [research](https://muse.jhu.edu/pub/1/article/901022) suggests that these suggestions are generally less accurate the greater their Levenshtein distance from the original word. When searching for replacement words, then, Logion filters model suggestions per a chosen Levenshtein distance.

On the Error Detection page, users may select a Levenshtein distance of 1, 2, or 3. Logion uses that value to filter the model's suggestions.
