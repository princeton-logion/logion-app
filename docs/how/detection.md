# Error Detection

## How to Use Logion Error Detection

First, navigate to to the error detection window by clicking "Error Detection" on the right-hand side of the main menu.

Inside the Error Detection window, there are three main features necessary to generate possible mis-transcribed words and potential replacements for a text.

On the upper-left of the window, select which Logion model you want to use from the drop-down menu. If this is your first time, we recommend beginning with "Base BERT". This has been trained on a wide selection of pre-modern Greek and is suitable for general error-detection recommendations.

The Levenshtein distance drop-down menu is to the right of the model selection menu. Levenshtein distance is a term from natural language processing that refers to the minimum number of alterations needed to transform one word into another. As such, the selected Levenshtein distance will here limit the model's suggestions to words that have that many character-letter differences from words in the original input text. We recommend starting with a Levenshtein distance of 1.

Once you have selected your preferred model, you may type/paste your text into the text area. Unlike, Logion's word prediction feature, do NOT use "?" to represent any missing words. Error detection results are better the less missing words the input text has. Input text that contains only Greek script and punctuation.

Then, click the blue "Detect Errors" button below the text area. Note the error detection process can take several minutes depending on one's local hardware. To read more on how hardware affects error detection processing speed, click [here](https://princeton-logion.github.io/logion-app/hardware/).

Logion will display error detection results below the "Detect Errors" button. Text is color-coded to signify each given word's likelihood of it having been mis-transcribed at some point in the work's textual history. Green means the word is unlikely to be mis-transcribed, red means the word is very likely to be mis-transcribed. To see what the model suggests as the correct word, click on a given word. Alternative word suggestions will be displayed on the right-hand side of the window. You may simply click on a different word to display that word's results.

## How Error Detection Works

COMING SOON