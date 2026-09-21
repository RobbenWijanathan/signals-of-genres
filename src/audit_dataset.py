from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

AUDIO_DIR = PROJECT_DIR / "dataset" / "fma_small"
TRACKS_CSV = PROJECT_DIR / "dataset" / "fma_metadata" / "tracks.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def get_audio_path(track_id: int) -> Path:
    track_id = f"{track_id:06d}"

    return (
        AUDIO_DIR
        / track_id[:3]
        / f"{track_id}.mp3"
    )


tracks = pd.read_csv(
    TRACKS_CSV,
    index_col=0,
    header=[0, 1],
)

small = tracks[
    tracks[("set", "subset")] == "small"
].copy()

audit = pd.DataFrame({
    "track_id": small.index.astype(int),
    "genre": small[("track", "genre_top")].values,
    "split": small[("set", "split")].values,
})

audit["audio_path"] = audit["track_id"].apply(get_audio_path)
audit["exists"] = audit["audio_path"].apply(Path.exists)

print(f"Expected tracks: {len(audit)}")
print(f"Audio files found: {audit['exists'].sum()}")
print(f"Missing files: {(~audit['exists']).sum()}")

print("\nTracks per genre:")
print(audit["genre"].value_counts().sort_index())

print("\nOfficial splits:")
print(audit["split"].value_counts())

print("\nGenre distribution by split:")
print(
    audit.groupby(["split", "genre"])
    .size()
    .unstack(fill_value=0)
)

missing = audit[~audit["exists"]]

if not missing.empty:
    print("\nFirst missing files:")
    print(missing.head(20))

audit.to_csv(
    OUTPUT_DIR / "dataset_audit.csv",
    index=False,
)