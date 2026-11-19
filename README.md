# Alpha PCA on Fashion MNIST

This CISC3024 Pattern Recognition experiment compares Φα-PCA and classical PCA in the low-dimensional classification task described in Figure 8 of the paper. The course implementation uses Fashion-MNIST's official split and a fixed, balanced subset of its training set. Its primary comparison is Φα-PCA with α = 0.75 against classical PCA with α = 1.0 at 32 components.

## Run the experiment

Run these commands from the project directory containing `source/`. Use Python 3.12.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r source/requirements.txt
.venv/bin/python source/download_data.py
.venv/bin/python source/run_experiment.py --pilot
.venv/bin/python source/run_experiment.py
.venv/bin/python source/plot_results.py
```

The downloader checks each archive against Zalando's published MD5 checksum. The environment is pinned to the tested versions. On Apple Silicon, the requirements select the CPython 3.12 OpenBLAS build of NumPy. By default, the experiment uses four compute threads. On Windows, use `.venv\Scripts\python.exe` for the commands after environment creation.

The pilot uses 1,200 training and 1,000 test images. It verifies the dimensions, sign-preserving power map, equivalence to classical PCA when α = 1, and test corruption. Its scores are in `results/pilot/`. The formal run uses 12,000 training and all 10,000 test images. It runs five α values and evaluates the first 8, 16 and 32 components of each fit.

## Protocol

- Sample 1,200 training images per class with NumPy's default random generator and seed 3024. Keep training images clean and scale the original bytes by 1/255. Use the official test set with the same initial scaling.
- For each α, apply the elementwise map `sign(x) * abs(x)**alpha`, fit centered randomized PCA with seed 3024, and apply the inverse power map to the projected scores. Fit logistic regression on the clean latent training data for each component count. The solver is L-BFGS with C = 1, maximum 1,000 iterations, and tolerance 0.0001.
- Evaluate the classifier on clean test images and three corrupted copies. In each copy, select pixels independently with probability 0.01, multiply the selected values by 10, and preserve their full range. The three seeds are 3025, 3026 and 3027. Every model sees the same test images and the same three masks.
- The reported drop in percentage points is clean accuracy minus the arithmetic mean of the three corrupted accuracies. The reported standard deviation summarizes only variation across masks on the fixed test set.

## Result files

| File | Contents |
| --- | --- |
| `results/checkpoint.json` | Protocol, versions, data fingerprints, mask diagnostics, and results saved after each configuration |
| `results/accuracy.csv` | Correct counts and accuracies for each clean or corrupted evaluation |
| `results/summary.csv` | Clean and mean corrupted accuracy, mask variability, accuracy drop, and model-fitting diagnostics |
| `report/accuracy-by-components.png` | Accuracy curves from the formal summary |

Repeat the same command to resume an interrupted run. The runner uses the checkpoint to skip finished configurations and recover the CSVs. Use a fresh directory through `--output-dir` to rerun from scratch. Run one process per output directory.

## Archived run

The final course run and its figure are included under reference-results/. A fresh run writes to results/ so the archived checkpoint is preserved.

Regenerate the figure from the archived table with this command.

```sh
.venv/bin/python source/plot_results.py --summary reference-results/summary.csv
```

## References

- Mussard, C., Briec, W., Condevaux, C., & Mussard, S. (2026). Principal component analysis in Φα-linear spaces. *Journal of Big Data*, 13, 128. [Theorem 1 and Figure 8](https://doi.org/10.1186/s40537-026-01473-x)
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST. [Official dataset](https://github.com/zalandoresearch/fashion-mnist)
