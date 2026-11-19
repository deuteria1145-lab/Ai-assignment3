"""Run a paired clean-versus-corrupted Fashion-MNIST classification benchmark."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import struct
from time import perf_counter
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits

from alpha_pca import AlphaPCA, signed_power


PROJECT = Path(__file__).resolve().parents[1]
TRAIN_SEED = 3024
MASK_SEEDS = (3025, 3026, 3027)
PCA_SEED = 3024
MASK_PROBABILITY = 0.01
BRIGHT_PIXEL_GAIN = 10
DATA_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
FULL_ALPHAS = (0.5, 0.75, 1.0, 1.25, 1.5)
FULL_COMPONENT_COUNTS = (8, 16, 32)
PRIMARY_COMPARISON = {"alpha_pca": 0.75, "baseline_alpha": 1.0, "components": 32}
CLASSIFIER_OPTIONS = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4}


def load_arrays(data_dir, train_per_class, test_per_class):
    for filename, expected_md5 in DATA_FILES.items():
        data_path = data_dir / filename
        actual = hashlib.md5(data_path.read_bytes()).hexdigest()
        if actual != expected_md5:
            raise ValueError(f"Checksum mismatch for {data_path}: {actual}")

    def read_split(prefix, expected_count):
        with gzip.open(data_dir / f"{prefix}-images-idx3-ubyte.gz", "rb") as stream:
            image_shape = struct.unpack(">IIII", stream.read(16))
            images = np.frombuffer(stream.read(), dtype=np.uint8)
        with gzip.open(data_dir / f"{prefix}-labels-idx1-ubyte.gz", "rb") as stream:
            label_shape = struct.unpack(">II", stream.read(8))
            labels = np.frombuffer(stream.read(), dtype=np.uint8)
        if image_shape != (2051, expected_count, 28, 28):
            raise ValueError(f"Unexpected {prefix} image header: {image_shape}")
        if label_shape != (2049, expected_count) or labels.shape != (expected_count,):
            raise ValueError(f"Unexpected {prefix} label dimensions: {label_shape}")
        if not np.array_equal(np.bincount(labels, minlength=10), [expected_count // 10] * 10):
            raise ValueError(f"Unexpected class counts in the {prefix} split.")
        return images.reshape(expected_count, 784), labels

    train_images, train_labels = read_split("train", 60000)
    test_images, test_labels = read_split("t10k", 10000)
    train_indices = balanced_indices(train_labels, train_per_class, TRAIN_SEED)
    test_indices = (
        np.arange(test_labels.size)
        if test_per_class is None
        else balanced_indices(test_labels, test_per_class, TRAIN_SEED)
    )
    return (
        train_images[train_indices].astype(np.float32) / np.float32(255),
        train_labels[train_indices],
        test_images[test_indices].astype(np.float32) / np.float32(255),
        test_labels[test_indices],
        train_indices,
        test_indices,
    )


def balanced_indices(labels, size_per_class, seed):
    rng = np.random.default_rng(seed)
    selected = [
        rng.choice(np.flatnonzero(labels == label), size_per_class, replace=False)
        for label in range(10)
    ]
    return np.sort(np.concatenate(selected))


def build_noisy_tests(clean):
    noisy = []
    diagnostics = []
    for seed in MASK_SEEDS:
        mask = np.random.default_rng(seed).random(clean.shape) < MASK_PROBABILITY
        corrupted = clean.copy()
        corrupted[mask] *= BRIGHT_PIXEL_GAIN
        noisy.append(corrupted)
        diagnostics.append({
            "seed": seed,
            "selected_pixels": int(mask.sum()),
            "changed_pixels": int(np.count_nonzero(clean != corrupted)),
            "total_pixels": clean.size,
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        })
    return noisy, diagnostics


def verify_pipeline(train, train_labels, clean, noisy, diagnostics):
    if train.shape != (train_labels.size, 784) or clean.shape[1] != 784:
        raise ValueError("Unexpected model input shape.")
    if not np.all(np.isfinite(train)) or not 0 <= train.min() <= train.max() <= 1:
        raise ValueError("Training images must contain clean pixels scaled into [0, 1].")
    if len({trial["mask_sha256"] for trial in diagnostics}) != len(MASK_SEEDS):
        raise ValueError("The independent trials must use distinct masks.")
    for corrupted, trial in zip(noisy, diagnostics):
        if corrupted.shape != clean.shape or not 0.009 <= (
            trial["selected_pixels"] / trial["total_pixels"]
        ) <= 0.011:
            raise ValueError("The corrupted test shape or selection rate is incorrect.")
        changed = corrupted != clean
        if np.any(corrupted[changed] != clean[changed] * BRIGHT_PIXEL_GAIN):
            raise ValueError("The corruption must multiply selected pixels without clipping.")
        if trial["changed_pixels"] == 0 or corrupted.max() <= 1:
            raise ValueError("The corruption must create bright out-of-range pixels.")

    samples, targets = train[:100], clean[:20]
    values = np.asarray([-4.0, -0.25, 0.0, 0.25, 4.0])
    np.testing.assert_allclose(signed_power(signed_power(values, 0.75), 1 / 0.75), values)
    model = AlphaPCA(alpha=1.0, n_components=8, random_state=PCA_SEED).fit(samples)
    reference = PCA(n_components=8, svd_solver="randomized", random_state=PCA_SEED).fit(samples)
    identity_scores = model.transform(targets)
    if identity_scores.shape != (20, 8):
        raise ValueError("Unexpected PCA score shape.")
    np.testing.assert_allclose(identity_scores, reference.transform(targets), rtol=1e-6)


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def tables_from_records(records):
    accuracy_rows, summary_rows = [], []
    for record in sorted(records, key=lambda row: (row["components"], row["alpha"])):
        common = {"alpha": record["alpha"], "components": record["components"]}
        measurements = record["measurements"]
        accuracy_rows.extend({**common, **measurement} for measurement in measurements)
        clean_accuracy = measurements[0]["accuracy_pct"]
        noisy_values = [value["accuracy_pct"] for value in measurements[1:]]
        noisy_mean = float(np.mean(noisy_values))
        summary_rows.append({
            **common,
            "clean_accuracy_pct": clean_accuracy,
            "corrupted_mean_accuracy_pct": round(noisy_mean, 6),
            "corrupted_sd_pp": round(float(np.std(noisy_values, ddof=1)), 6),
            "clean_to_corrupted_drop_pp": round(clean_accuracy - noisy_mean, 6),
            "pca_fit_seconds": record["pca_fit_seconds"],
            "classifier_fit_seconds": record["classifier_fit_seconds"],
            "classifier_iterations": record["classifier_iterations"],
            "classifier_converged": record["classifier_converged"],
        })
    return accuracy_rows, summary_rows


def save_tables(output_dir, records):
    accuracy_rows, summary_rows = tables_from_records(records)
    atomic_csv(output_dir / "accuracy.csv", accuracy_rows)
    atomic_csv(output_dir / "summary.csv", summary_rows)


def evaluate_classifier(classifier, tests, labels):
    measurements = []
    for seed, test_scores in zip((None, *MASK_SEEDS), tests):
        correct = int(np.count_nonzero(classifier.predict(test_scores) == labels))
        measurements.append({
            "condition": "clean" if seed is None else "corrupted",
            "mask_seed": "" if seed is None else seed,
            "correct": correct,
            "total": labels.size,
            "accuracy_pct": round(100 * correct / labels.size, 6),
        })
    return measurements


def make_metadata(args, alphas, component_counts, train_labels, test_labels,
                  train_indices, test_indices, diagnostic_rows):
    return {
        "mode": "pilot" if args.pilot else "formal",
        "dataset": "Fashion-MNIST",
        "dataset_md5": DATA_FILES,
        "training_examples": int(train_labels.size),
        "test_examples": int(test_labels.size),
        "training_counts_by_class": np.bincount(train_labels, minlength=10).tolist(),
        "test_counts_by_class": np.bincount(test_labels, minlength=10).tolist(),
        "training_indices_sha256": hashlib.sha256(train_indices.astype("<i8").tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(test_indices.astype("<i8").tobytes()).hexdigest(),
        "training_subset_seed": TRAIN_SEED,
        "training_input": "clean float32 pixels scaled by 1/255",
        "mask_probability": MASK_PROBABILITY,
        "selected_pixel_multiplier": BRIGHT_PIXEL_GAIN,
        "clip_corrupted_pixels": False,
        "mask_diagnostics": diagnostic_rows,
        "alphas": list(alphas),
        "components": list(component_counts),
        "pca": {"solver": "randomized", "random_state": PCA_SEED, "whiten": False},
        "classifier": CLASSIFIER_OPTIONS.copy(),
        "primary_comparison": None if args.pilot else PRIMARY_COMPARISON.copy(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "platform": platform.platform(),
            "threadpools": [
                {
                    "internal_api": item["internal_api"],
                    "num_threads": item["num_threads"],
                    "version": item["version"],
                }
                for item in threadpool_info()
            ],
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("alpha_pca.py"))
        },
    }


def run(args):
    alphas = (0.75, 1.0) if args.pilot else FULL_ALPHAS
    component_counts = (8, 16) if args.pilot else FULL_COMPONENT_COUNTS
    train, train_labels, clean, test_labels, train_indices, test_indices = load_arrays(
        args.data_dir, train_per_class=120 if args.pilot else 1200,
        test_per_class=100 if args.pilot else None,
    )
    noisy, diagnostic_rows = build_noisy_tests(clean)
    verify_pipeline(train, train_labels, clean, noisy, diagnostic_rows)
    print(f"Verified {len(train):,} train and {len(clean):,} test images, alpha=1 identity, "
          "and three paired corruption masks.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.json"
    metadata = make_metadata(args, alphas, component_counts, train_labels, test_labels,
                             train_indices, test_indices, diagnostic_rows)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["run"] != metadata:
            raise ValueError("Checkpoint configuration has changed. Use a fresh output directory.")
    else:
        checkpoint = {"run": metadata, "records": []}
        atomic_text(checkpoint_path, json.dumps(checkpoint, indent=2) + "\n")
    records = checkpoint["records"]
    completed = {(record["alpha"], record["components"]) for record in records}
    save_tables(args.output_dir, records)

    for alpha in alphas:
        pending = [k for k in component_counts if (alpha, k) not in completed]
        if not pending:
            continue
        start = perf_counter()
        projector = AlphaPCA(alpha, max(component_counts), PCA_SEED).fit(train)
        pca_seconds = round(perf_counter() - start, 6)
        train_scores = projector.transform(train).astype(np.float64)
        test_scores = [projector.transform(images).astype(np.float64) for images in (clean, *noisy)]
        if not all(np.all(np.isfinite(scores)) for scores in (train_scores, *test_scores)):
            raise ValueError(f"Non-finite latent scores for alpha {alpha}.")

        for k in pending:
            classifier = LogisticRegression(**CLASSIFIER_OPTIONS)
            start = perf_counter()
            with warnings.catch_warnings(record=True) as classifier_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(train_scores[:, :k], train_labels)
            classifier_seconds = round(perf_counter() - start, 6)
            if not np.all(np.isfinite(classifier.coef_)) or not np.all(
                np.isfinite(classifier.intercept_)
            ):
                raise ValueError(f"Non-finite classifier parameters for alpha {alpha}, k={k}.")
            converged = all(
                not issubclass(item.category, ConvergenceWarning) for item in classifier_warnings
            )
            for item in classifier_warnings:
                print(f"Classifier {item.category.__name__} at alpha={alpha:g}, k={k}: "
                      f"{item.message}", flush=True)
            measurements = evaluate_classifier(
                classifier, [scores[:, :k] for scores in test_scores], test_labels
            )
            records.append({
                "alpha": alpha,
                "components": k,
                "measurements": measurements,
                "pca_fit_seconds": pca_seconds,
                "classifier_fit_seconds": classifier_seconds,
                "classifier_iterations": int(np.max(classifier.n_iter_)),
                "classifier_converged": converged,
            })
            atomic_text(checkpoint_path, json.dumps(checkpoint, indent=2) + "\n")
            save_tables(args.output_dir, records)
            corrupted_mean = np.mean([value["accuracy_pct"] for value in measurements[1:]])
            print(f"Saved {len(records)}/{len(alphas) * len(component_counts)} "
                  f"alpha={alpha:g}, k={k}: clean {measurements[0]['accuracy_pct']:.2f}%, "
                  f"corrupted {corrupted_mean:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Use the smaller verified pilot.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4, help="BLAS and OpenMP thread limit.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.output_dir is None:
        args.output_dir = PROJECT / "results" / "pilot" if args.pilot else PROJECT / "results"
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
