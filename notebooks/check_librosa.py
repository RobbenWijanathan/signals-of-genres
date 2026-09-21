import librosa

path = "dataset/fma_small/000/000002.mp3"

y, sr = librosa.load(
    path,
    sr=22050,
    duration=30,
)

print("Sample rate:", sr)
print("Samples:", len(y))
print("Duration:", len(y) / sr)