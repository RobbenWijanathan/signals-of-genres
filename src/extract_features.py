from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
AUDIT_CSV = PROJECT_DIR / "outputs" / "dataset_audit.csv"
OUTPUT_CSV = PROJECT_DIR / "outputs" / "features.csv"

SAMPLE_PER_GENRE = 5


def add_stats(output, name, values):
    output[f"{name}_mean"] = float(np.mean(values))
    output[f"{name}_std"] = float(np.std(values))


def extract_features(audio_path):
    y, sr = librosa.load(
        audio_path,
        sr=22050,
        mono=True,
        duration=30,
    )

    features = {}

    tempo = librosa.feature.tempo(y=y, sr=sr)
    features["tempo"] = float(np.asarray(tempo).reshape(-1)[0])

    add_stats(features, "rms", librosa.feature.rms(y=y)[0])

    add_stats(
        features,
        "zcr",
        librosa.feature.zero_crossing_rate(y)[0],
    )

    add_stats(
        features,
        "centroid",
        librosa.feature.spectral_centroid(y=y, sr=sr)[0],
    )

    add_stats(
        features,
        "bandwidth",
        librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
    )

    add_stats(
        features,
        "rolloff",
        librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
    )

    add_stats(
        features,
        "flatness",
        librosa.feature.spectral_flatness(y=y)[0],
    )

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    add_stats(features, "chroma", chroma)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
    )

    for i in range(13):
        add_stats(
            features,
            f"mfcc_{i + 1}",
            mfcc[i],
        )

    return features


audit = pd.read_csv(AUDIT_CSV)
audit = audit[audit["exists"]].copy()

sample = audit.copy()

rows = []
errors = []

for row in tqdm(
    sample.itertuples(index=False),
    total=len(sample),
    desc="Extracting features",
):
    try:
        features = extract_features(row.audio_path)

        features.update({
            "track_id": row.track_id,
            "genre": row.genre,
            "split": row.split,
        })

        rows.append(features)

        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    except Exception as error:
        errors.append({
            "track_id": row.track_id,
            "error": str(error),
        })

features_df = pd.DataFrame(rows)
features_df.to_csv(OUTPUT_CSV, index=False)

print("\nOutput shape:", features_df.shape)
print("Failed tracks:", len(errors))
print("Missing values:", features_df.isna().sum().sum())
print("\nTracks per genre:")
print(features_df["genre"].value_counts())

if errors:
    pd.DataFrame(errors).to_csv(
        PROJECT_DIR / "outputs" / "extraction_errors.csv",
        index=False,
    )