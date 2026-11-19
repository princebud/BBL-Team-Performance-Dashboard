# Actual data folder

Place the Cricsheet Big Bash League JSON archive here. Download it from https://cricsheet.org/downloads/ and check the current licence/credit terms.

The raw archive and generated real-data CSVs are excluded from Git by `.gitignore`. Use:

```bash
python build_dataset.py data/bbl_json.zip
python data_processing.py
```

Synthetic example CSVs live in `demo_data/` and are safe for testing the pipeline; they are not real match data.
