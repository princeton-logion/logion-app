
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


# handle papyrology/orthography markers
_EDITORIAL_MARKS = {0x0323, 0x0332}
_APOSTROPHE_FOLD = str.maketrans(
    {"\u2019": "\u02bc", "\u1fbd": "\u02bc", "'": "\u02bc"})
_COMBINING_RANGE = "\u0300-\u0344\u0346-\u036f"
_ADSCRIPT_IOTA_RE = re.compile(
    f"([\u03b7\u03c9][{_COMBINING_RANGE}]*)\u03b9(?![{_COMBINING_RANGE}]*\u0308)")
 
 
def normalize_grc_input(text: str) -> str:
    """
    Fold edition/papyrological orthography to training preprocessing:
        - rmv editorial combining marks (sublinear dots)
        - unify elision-marker variants -> \u02bc
        - adscript iota -> subscript NFD level
 
    Parameters:
        text (str) -- txt to normalize
 
    Returns:
        normalized text
    """
    t = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if ord(c) not in _EDITORIAL_MARKS
    )
    t = _ADSCRIPT_IOTA_RE.sub("\\1\u0345", t)
    t = unicodedata.normalize("NFC", t)
    return t.translate(_APOSTROPHE_FOLD)
