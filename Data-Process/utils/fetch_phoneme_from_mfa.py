import argparse

# 建立语言和数字的映射
enum = {
    'english': 0,
    'german': 1,
    'russian': 2,
    'french': 3,
    'spanish': 4,
    'italian': 5,
    'japanese': 6,
    'korean': 7,
}

ENG_DICT_PATH = './mfa_dict/english_mfa.dict'
GER_DICT_PATH = './mfa_dict/german_mfa.dict'
RUS_DICT_PATH = './mfa_dict/russian_mfa.dict'
FRE_DICT_PATH = './mfa_dict/french.dict'
SPA_DICT_PATH = './mfa_dict/spanish_mfa.dict'
ITA_DICT_PATH = './mfa_dict/italian.dict'
JPN_DICT_PATH = './mfa_dict/japanese.dict'
KOR_DICT_PATH = './mfa_dict/korean_mfa.dict'

DICT_PATH_LIST = [ENG_DICT_PATH, GER_DICT_PATH, RUS_DICT_PATH, FRE_DICT_PATH, SPA_DICT_PATH, ITA_DICT_PATH, JPN_DICT_PATH, KOR_DICT_PATH]

def fetch_phonemes(word: str, language: int = 0) -> str:
    """
    Fetches the phonemes for a given word from a dictionary file, ensuring correct parsing of the phonemes.
    
    Args:
    word (str): The word to look up.
    dict_file_path (str): Path to the dictionary file.
    
    Returns:
    str: The phonemes corresponding to the word, if found. Otherwise, returns None.
    """
    dict_file_path = DICT_PATH_LIST[language]  
    # choose the corresponding dictionary file based on the language
    with open(dict_file_path, 'r') as file:
        for line in file:
            parts = line.split()
            dict_word = parts[0]
            if dict_word == word:
                # Filter out numeric parts to isolate phonemes
                phonemes = [part for part in parts if not part.replace('.', '', 1).isdigit()]
                return ' '.join(phonemes[1:])  # Skip the first element as it's the word itself
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch phonemes for a given word from a dictionary file")
    parser.add_argument("--word", type=str, help="The word to look up in the dictionary")
    parser.add_argument("--language", type=str, help="The language of the dictionary", default='english')
    args = parser.parse_args()
    language = enum[args.language]
    phonemes = fetch_phonemes(args.word, language)
    if phonemes:
        print(f"The phonemes for '{args.word}' are: {phonemes}")
    else:
        print("Word not found in the dictionary.")