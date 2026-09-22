# signals-of-genres

The Signals of Music Genres: exploring audio features from FMA Small.

## After feature extraction

The exploration checks `outputs/features.csv` against `outputs/dataset_audit.csv`
and, when present, `outputs/extraction_errors.csv`. Keep these files together.
It checks missing/infinite values, duplicate IDs, genre/split consistency and
whether failed tracks account for every missing extraction. All feature
comparisons and correlations use only the official **training** split.

### Run the notebook in VS Code

1. In your project terminal, install the analysis dependencies into the Python
   environment you intend to use:

   ```powershell
   python -m pip install -r requirements-eda.txt
   ```

2. Open `notebooks/01_feature_exploration.ipynb`.
3. Click **Select Kernel** and choose that Python environment. VS Code may prompt
   you to install its Python/Jupyter extensions if they are not already available.
4. Click **Run All**, or run each cell in order to read the explanations as you go.

The notebook works when started from the project root or its `notebooks` folder.
It does not extract audio again.

### Run without a notebook

From the project folder:

```powershell
python src/explore_features.py
```

Close each plot window to continue if your plotting backend opens windows. To
save plots without opening windows in PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
python src/explore_features.py
Remove-Item Env:MPLBACKEND
```

Both versions produce `outputs/eda/summary.md` and four PNG figures: genre
distributions, individual feature associations with genre, standardized genre
profiles, and feature correlations. Reruns replace these analysis outputs.
The notebook and script contain the same analysis code.

Feature rankings use descriptive eta-squared, not classification accuracy.
After reviewing the plots, the next stage is a baseline classifier: train using
the training split, choose settings using validation, and reserve test for the
final evaluation.
