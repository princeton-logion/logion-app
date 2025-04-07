# Error detection

Logion's error detection feature is aims to suggest the likelihood that each word in a given sequence is an error, i.e. was mis-transcribed at some point in the textual history. It further offers suggestions as to what the "correct" word may be.

## How to use Logion error detection

From the main menu, go to to to the error detection window by clicking **Error detection** on the right-hand side of the main menu. Once you are in the Error Detection page, follow these steps to generate an error report for your text.

1. Select a model from the drop-down menu in the upper-left of the window. If this is your first time, we recommend beginning with **Base BERT**. This has been trained on a wide selection of pre-modern Greek and is suitable for general error-detection recommendations.

1. Select a Levenshtein distance from the drop-down menu to the right of the model selection menu. We recommend starting with a Levenshtein distance of **1**. For more on what is Levenshtein distance, see our [explainer](../explainers/lev-dist.md).

1. Type/paste your text into the text area. Unlike, Logion's word prediction feature, do NOT use **?** to represent any missing words. Enter text comprised of only Greek characters and other punctuation marks.

1. Click the blue **Detect Errors** button below the text area. Note the error detection process can take several minutes depending on one's local hardware. To read more on how hardware affects error detection processing speed, click [here](../hardware.md).

Results are displayed below the blue **Detect Errors** button.

## How to read error reports

Text is color-coded to signify each given word's likelihood of it having been mis-transcribed at some point in the textual history. Text is colored on a gradient of green-yellow-orange-red. Green means the word is unlikely to be mis-transcribed; red means the word is very likely to be mis-transcribed. To see what the model suggests as a potential replacement word, click on a given word. Replacement word suggestions are displayed on the right-hand side of the window beside the model's projected (chance-confidence ratio)[../explainers/ccr.md] for that word-pair. To see a different word's results, simply click on that different word.
