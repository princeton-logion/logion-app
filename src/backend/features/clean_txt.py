
import unicodedata
import re

SPECIAL_CHARACTERS = r'<>«»⟨⟩𐅸\{\}\[\]※⁘^+=≈∽…₍₎–⊗#$%&\|‖‡†+§‖⟦⟧¹±~#@¡¿½*⸏?!𐅸⸏¾★⁋*𐅸←↑→−∙•∞∥∴∵∻჻᠅✣✤✳❈⟀⁖⁙⁚⁛⁜༶⸭〈〉⟪⟫⎛⎝⎞⎠⎧⎨⎩⎪⎫⎬⎭⌜⌝⌞⌟\\\/""„\"ʹ´`´ʹ´ʼ′´´´῾᾿`῀¨¨᾿῾𐅻𐅵‹ϙ͵›𐆄⏑𐅷𐅶ϡˈϟ×♃᾽⁝𐆊□‵𐆃‚‛⩚⁄´⸐˙⸖𐆆𐆂ϝ⏒⏔⸓☾𝈓⸕⏓𐅼`𐆅⸎☩☉♀𝈶𝈳☍𝈱○᾿⏕♄♂≌𝈩𝈈ͻ☿𝈿∠𝈸𝉀♌♏♎♑𐆈♈♋𝈍♉♊♐♓𝈕♍♒℧⁞⋮ˋˆ𝈖𐄑𝈒⸒𝈚𝈪𝈷𝈏⏖𝈎ˊ𝈛𝈅𝈗𝈆῾¸𝈔𝈨𝈲𝈹𝈑𝈜☽፠𝈰𝈁𝈵𐅽𝈞𝈀𝈉𝈌𝈡𝈥𝈬𝈋𝈙𝈂𝈤𝈮𝈾𝉁𝈺𝈴⸑ͼ𐄒𐅄𝈃𝈝𝈟𝈯𝈭𝈐𝈊𝈇𝈘𝈣𝈽ʽ⩫ͽ𐆉𐅀⸔𐅹𝈦𝈢𝈠𝈄𐅾ϻꙩ¨𐅃☋𐅆𐅅☌⟘⟁÷﻿\—‧···;;:,""ϡϟϝ'

def clean_input_txt(text: str) -> str:
    """
    Clean txt by:
        - rmv special characters
        - rmv accents + diacritics
    
    Parameters:
        text: txt to clean
    
    Returns:
        cleaned text
    """
    # preserve [MASK]s
    mask_placeholder = "MASK_TOKEN_PLACEHOLDER"
    text = text.replace("[MASK]", mask_placeholder)
    
    # rmv special chars
    translator = str.maketrans(SPECIAL_CHARACTERS, ' ' * len(SPECIAL_CHARACTERS))
    text = text.translate(translator)
    
    # rmv digits
    text = re.sub(r'\d+', ' ', text)
    
    # lowercase
    text = text.lower()
    # except mask_placeholder
    mask_placeholder = mask_placeholder.lower()
    
    # Step 5: Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # restore [MASK]s
    text = text.replace(mask_placeholder, "[MASK]")

    # rmv accents + diacritics
    nfkd_form = unicodedata.normalize('NFKD', text)

    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    