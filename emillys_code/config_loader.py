import pandas as pd 
from pathlib import Path

def load_config(language, key):
    config_path = Path("C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/language_config.json")
    config_df = pd.read_json(config_path)
    config_df.set_index("Language", inplace=True)

    try:
        lang_cfg = config_df.loc[language]
        result = lang_cfg[key]
    except KeyError:
        print(f"Key '{key}' not found, either {language} or {key} is not supported.")
        return
    return result
