# Dataset Card: STScI Exoplanet Candidate Target List (xCTL v08.01)

## Dataset Overview
- **Name:** Exoplanet Candidate Target List (xCTL)
- **Version:** v08.01
- **Source:** NASA / STScI MAST Archive
- **Format:** CSV (497 MB)
- **Primary Use:** Prioritizing TESS targets for Exoplanet Transit Detection.

## Dataset Characteristics
The xCTL is a specialized subset of the massive TESS Input Catalog (TIC). It filters billions of point sources down to targets that are statistically and physically capable of hosting detectable transiting exoplanets.

### Included Features
- `TIC_ID`: Unique STScI identifier.
- `Tmag`: TESS magnitude (brightness).
- `Teff`: Effective stellar temperature (Kelvin).
- `mass`: Stellar mass (relative to solar).
- `rad`: Stellar radius (relative to solar).
- `lum`: Stellar luminosity.
- `d`: Distance to target (parsecs).
- `prioriy`: Continuous viewing zone priority metric.

## Filtering & Curation Methodology (AstroVerse Pipeline)
For the AstroVerse Phase 1 MVP, the 497 MB CSV is ingested and strictly filtered using the following physics bounds to isolate the highest-yield candidates:
1. **Dwarf Stars Only:** `rad < 2.0` (Excludes giant stars where planetary transits are too shallow to detect).
2. **Temperature Bounds:** `3000K < Teff < 7000K` (M, K, G, and late F-type stars).
3. **Brightness Constraints:** `Tmag < 13.0` (Ensures sufficient photon count for high Signal-to-Noise Ratio).

## Known Limitations & Biases
- **Binary Contamination:** The xCTL still contains unresolved grazing eclipsing binaries. The EvoNex `Physics MLP` is specifically trained to identify these false positives.
- **M-Dwarf Bias:** The catalog naturally favors M-dwarfs due to the deeper transit signatures relative to stellar size.
- **Crowding:** High-density galactic plane regions may suffer from pixel bleeding (background flux), inflating the apparent noise in the light curves.

## Ethical & Scientific Considerations
- **Open Access:** This dataset is public domain under NASA/STScI guidelines.
- **Reproducibility:** All AstroVerse inferences map back directly to the immutable TIC IDs provided in this catalog, ensuring 100% verifiability by third-party astronomers.
