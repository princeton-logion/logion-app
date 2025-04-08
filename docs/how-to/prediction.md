# Word prediction

Logion's word prediction feature predicts words that may be missing in a text. The feature works best for filling gaps that likely span no more than one word.

## How to use Logion word prediction

From the main menu, go to to to the word prediction window by clicking **Word prediction** on the left-hand side of the main menu. Once you are in the Word Prediction page, follow these steps to generate an single-word predictions for gaps in your text.

1. Select a model from the drop-down menu in the upper-left of the window. If this is your first time, we recommend beginning with **Base BERT**. [This model](https://huggingface.co/princeton-logion/LOGION-50k_wordpiece) is trained on a wide selection of premodern Greek and is suitable for general gap-filling.

1. Type/paste your text into the text area. Use **?** to represent any missing words for which you want to generate suggestions. Enter text comprised of only Greek characters and other punctuation marks.

1. Click the blue **Predict** button below the text area. This may take several seconds, particularly if this is your first time using the app or a given model. When finished, Logion displays word suggestions with their assigned probabilities to the right of the text box.