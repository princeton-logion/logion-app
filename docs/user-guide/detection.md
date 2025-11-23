# Error detection

Logion's error detection feature predicts the likelihood that each word in a given sequence is an error. Here, *error* means a word that has been mistranscribed at some point in the textual history. Logion further offers suggestions as to what a possible "correct" word may be.

## How to use Logion error detection

From the main menu, go to to to the error detection window by clicking **Error detection** on the right-hand side of the main menu. Once you are in the Error Detection page, follow these steps to generate an error report for your text.

1. **Select** a model from the drop-down menu in the upper-left of the window. If this is your first time, we recommend beginning with **Base BERT**. [This model](https://huggingface.co/princeton-logion/LOGION-50k_wordpiece) is trained on a wide selection of premodern Greek and is suitable for general error detection.

1. **Select** a Levenshtein distance from the drop-down menu to the right of the model selection menu. We recommend starting with a Levenshtein distance of **1**. To learn more about Levenshtein distance, see Logion's [explainer](../explainers/lev-dist.md).

1. **Type** your text into the text area. Unlike, Logion's gap prediction feature, don't use "**-**" to represent any missing words.  For best results, enter only text comprised of Greek characters and periods.

1. **Click** the blue **Detect Errors** button below the text area. Note the error detection process can take several minutes depending on one's local hardware. To read more on how hardware affects processing speed, see Logion's [hardware guide](../hardware.md).

Logion displays results below the blue **Detect Errors** button.

## How to read error reports

Text is color-coded to signify each given word's likelihood of it having been mistranscribed at some point in the textual history. Text is colored on a gradient of green-yellow-orange-red. Green means the word is unlikely to be mistranscribed; red means the word is very likely to be mistranscribed. To see what the model suggests as a potential replacement word, click a given word. Replacement word suggestions are displayed on the right-hand side of the window beside the model's projected [chance-confidence ratio](../explainers/ccr.md) for that word pair. To see a different word's results, simply click that different word.
