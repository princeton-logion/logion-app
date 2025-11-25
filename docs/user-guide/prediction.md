# Gap prediction

Logion's word and character prediction features predict missing parts of a text. Per their names, character prediction proposes missing text at the character level; word prediction proposes missing text at the word level. While character prediction can be used to fill whole or partial words, word prediction only proposes whole words.

## How to use Logion gap prediction

Navigate to either the Word Prediction or Character Prediction page by clicking its respective button on the main menu. Follow these steps to generate missing text predictions for gaps in your text:

1. **Select** a model from the drop-down menu in the upper-left of the window.
    - Only *Char* models may be used for character prediction; only *BERT* models may be used for word prediction.

1. **Type** your text into the text area. Use "**-**" to represent any missing words or characters (depending on which feature you are using). For best results, enter text comprised of only Greek characters and periods.

1. **Click** the blue **Predict** button below the text area. This may take several seconds, particularly if this is your first time using the app or a given model. When finished, 

Logion displays results below the blue **Predict** button. Proposed missing text is shaded deep purple and surround by square brackets **[]**. Logion also displays suggested words/characters with their assigned probabilities to the right of the text box. Users may need to click a word or character in the "Restored text" window to display this box.
