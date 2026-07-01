# Model Card: EvoMoE (EvoNex Mixture-of-Experts)

## Model Details
- **Architecture:** Explainable Mixture-of-Experts (MoE)
- **Framework:** PyTorch 2.1+
- **Input Tensors:** 
  1. `Lightcurve Tensor`: Shape `(Batch, 1, 2000)` - Normalized flux time-series.
  2. `Stellar Features Tensor`: Shape `(Batch, 13)` - Astrophysical constraints (Mass, Radius, Teff, etc.)
- **Output:** Binary Classification (1 = Exoplanet, 0 = False Positive) & Expert Routing Weights.

## Intended Use
EvoMoE is designed to act as an automated triaging agent for NASA TESS telemetry. It filters out astrophysical false positives (e.g., eclipsing binaries, instrumental noise) and highlights high-probability exoplanet candidates for human follow-up.

## Architecture Breakdown (The 3 Experts)
Unlike monolithic models (ResNet/ViT), EvoMoE dynamically routes inputs to specialized subnetworks:
1. **The Local Expert (Multi-Scale CNN):** Analyzes the morphological shape of the transit dip (U-shape vs V-shape).
2. **The Global Expert (Temporal Transformer):** Scans the entire 30-day sequence for complex orbital periodicity and out-of-transit variations.
3. **The Stellar Expert (Physics MLP):** Validates the transit depth against the physical limits of the host star's radius and mass using the TIC metadata.

## Benchmarks & Performance
*Note: Evaluated on the curated 100-target holdout set.*
- **Precision:** 0.942
- **Recall:** 0.961
- **F1-Score:** 0.951
- **ROC-AUC:** 0.978
- **Inference Latency:** <45ms per target (CPU), <5ms per target (GPU).

## Known Limitations & Failure Modes
1. **Grazing Eclipsing Binaries:** Highly eccentric grazing binaries can perfectly mimic planetary U-shape transits. The model relies heavily on the `Physics MLP` to reject these, but it may fail if the TIC metadata contains high-margin errors.
2. **Ultra-Long Period Planets:** Planets with orbital periods > 15 days will only transit once per TESS sector. The `Transformer` expert struggles to find a rhythm from a single dip, heavily shifting the routing burden to the `CNN`.
3. **Multi-Planet Systems:** Complex overlapping transits (Transit Timing Variations) can confuse the rhythm analysis, leading to lower confidence scores.

## Ethical Considerations
Automated astronomical detection pipelines dictate where massive observatory resources (like JWST) are pointed. Over-trusting AI predictions without explainability can lead to wasted telescope time. EvoMoE mitigates this by exposing the internal routing weights (Explainability Matrix), forcing the AI to justify *why* it made a prediction.
