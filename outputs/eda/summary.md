# Feature exploration report

Sources: outputs/features.csv, outputs/dataset_audit.csv, and outputs/extraction_errors.csv (if present).

- Extracted tracks: 7,994; failed tracks: 6.
- Audio features: 41; missing cells: 0; infinite values: 0.
- Training tracks used for feature exploration: 6,394.
- All integrity checks passed; every missing attempted track is accounted for in the error log.

## Track counts

```
split          training  validation  test  total
genre                                           
Electronic          799         100   100    999
Experimental        799         100   100    999
Folk                800         100   100   1000
Hip-Hop             797         100   100    997
Instrumental        800         100   100   1000
International       800         100   100   1000
Pop                 800         100   100   1000
Rock                799         100   100    999
```

## Failed tracks

```
 track_id        genre    split                                                                                                                                                                                                        error
    98565      Hip-Hop training                                                                                                                                                                                  Unspecified internal error.
    98567      Hip-Hop training                                                                                                                                                                                  Unspecified internal error.
    98569      Hip-Hop training                                                                                                                                                                                  Unspecified internal error.
    99134   Electronic training Error opening "C:\\Users\\Robben's Laptop\\OneDrive\\Documents\\Personal Projects\\signals-of-genres\\dataset\\fma_small\\099\\099134.mp3": File does not exist or is not a regular file (possibly a pipe?).
   108925         Rock training Error opening "C:\\Users\\Robben's Laptop\\OneDrive\\Documents\\Personal Projects\\signals-of-genres\\dataset\\fma_small\\108\\108925.mp3": File does not exist or is not a regular file (possibly a pipe?).
   133297 Experimental training Error opening "C:\\Users\\Robben's Laptop\\OneDrive\\Documents\\Personal Projects\\signals-of-genres\\dataset\\fma_small\\133\\133297.mp3": File does not exist or is not a regular file (possibly a pipe?).
```

## Largest differences in genre means

Eta-squared is a descriptive training-set association, not predictive accuracy.

```
                eta_squared
bandwidth_mean       0.2518
rolloff_mean         0.2409
chroma_mean          0.2401
centroid_std         0.2363
mfcc_1_mean          0.2359
centroid_mean        0.2167
mfcc_2_mean          0.1951
rms_std              0.1920
rolloff_std          0.1865
zcr_std              0.1813
rms_mean             0.1369
flatness_std         0.1317
mfcc_12_mean         0.1277
mfcc_5_std           0.1233
mfcc_10_mean         0.1228
```

## Strong feature correlations

Pairs with absolute Pearson correlation >= 0.90; training tracks only.

```
     feature_a    feature_b  correlation  absolute_correlation
 centroid_mean rolloff_mean       0.9725                0.9725
bandwidth_mean rolloff_mean       0.9431                0.9431
  centroid_std  rolloff_std       0.9007                0.9007
```

## Interpretation and next step

Compare the boxplots for both median differences and overlap. Review correlated pairs before deciding whether to remove features.
Train a baseline with preprocessing fitted only on training data; use validation for model choices and reserve test for final evaluation.
These observations describe this dataset and extraction method; they do not establish universal properties of genres.
