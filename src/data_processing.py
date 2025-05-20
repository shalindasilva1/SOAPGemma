from pathlib import Path
from datasets import load_dataset
import pandas as pd
import re

# Define paths using pathlib for easier directory creation
DATA_PATH = Path("data")
OMI_PATH_processed = DATA_PATH / "processed" / "omi-health"
OMI_PATH_raw = DATA_PATH / "raw" / "omi-health"
OMI_PATH_processed.mkdir(parents=True, exist_ok=True)

# Load the dataset
ds_omi_health = load_dataset("omi-health/medical-dialogue-to-soap-summary")

# Save the dataset to disk
ds_omi_health.save_to_disk(OMI_PATH_raw)

# Event tags extraction
def extract_dialogue_tags(dialogue):
    if pd.isna(dialogue):
        return []

    # Regex to find text enclosed in ( ) or [ ] that appears on its own line.
    # \n : Matches a newline character.
    # \s* : Matches any whitespace character (space, tab, newline, etc.) zero or more times.
    # ( : Start of a capturing group (this is what re.findall will return).
    #   \(.*?\) : Matches anything inside literal parentheses (non-greedy).
    #   | : OR operator.
    #   \[.*?\] : Matches anything inside literal square brackets (non-greedy).
    # ) : End of the capturing group.
    # \s* : Matches any whitespace character after the tag on the same line.
    # (?:\n|$) : Non-capturing group that matches either a newline character or the end of the string.
    # This ensures the tag is essentially on a line by itself.
    pattern = r"\n\s*(\(.*?\)|\[.*?\])\s*(?:\n|$)"
    tags = re.findall(pattern, str(dialogue))
    # convert list to a comma cooperated text
    text = str(tags)
    return text
def generate_tags_col(df, dialogue = "dialogue"):
    tags_df = df['dialogue'].apply(extract_dialogue_tags).apply(pd.Series)
    return_df = pd.concat([df, tags_df], axis=1)
    return_df = return_df.rename(columns={
        0: 'event_tags'
    })
    return return_df

if 'train' in ds_omi_health:
    train_df_omi = ds_omi_health['train'].to_pandas()

if 'validation' in ds_omi_health:
    val_df_omi = ds_omi_health['validation'].to_pandas()

if 'test' in ds_omi_health:
    test_df_omi = ds_omi_health['test'].to_pandas()

train_df_omi = generate_tags_col(ds_omi_health['train'].to_pandas())
val_df_omi = generate_tags_col(ds_omi_health['validation'].to_pandas())
test_df_omi = generate_tags_col(ds_omi_health['test'].to_pandas())

train_df_omi.to_csv(OMI_PATH_processed / 'train_v1.csv', index=False)
val_df_omi.to_csv(OMI_PATH_processed / 'val_v1.csv', index=False)
test_df_omi.to_csv(OMI_PATH_processed / 'test_v1.csv', index=False)