# EvoNex: Official Presentation & Explanation Guide
**Your Master Script for Hackathon Judges and Academic Peers**

This document is your complete "cheat sheet." It contains exactly how to explain the architecture in plain English, the scientific justification for every component, and the exact steps to follow during a live demonstration.

---

## Part 1: The Problem (The Hook)
*When you start your presentation, outline the problem clearly.*

**What to say:**
> "The NASA TESS telescope stares at stars and looks for tiny dips in light, hoping those dips are planets. But there's a huge problem: instrument glitches, camera noise, and binary star systems also cause dips in light. Furthermore, processing TESS data is agonizingly slow—downloading target pixel files can mean dealing with tens of gigabytes of CSVs. 
>
> To solve this, most teams build a standard Convolutional Neural Network (CNN). But standard CNNs are a 'black box.' They don't understand the physical laws of the universe. We wanted to build something better. We built **EvoNex**."

---

## Part 2: Explaining the Architecture
*This is the core of your project. Break down the EvoMoE (Evolutionary Mixture-of-Experts) into these four simple pieces.*

### 1. The Local Expert (Multi-Scale CNN)
*   **The Metaphor:** The "Shape Detector."
*   **How to explain it:** "Our first AI branch focuses entirely on the exact shape of the transit dip. But instead of one lens, we use a 'Multi-Scale' approach. It processes the light curve through three different sized filters simultaneously. Small filters catch the sharp, V-shaped dips of binary stars, while large filters catch the long, flat U-shaped dips of real planets."

### 2. The Global Expert (Temporal Transformer)
*   **The Metaphor:** The "Rhythm Detector."
*   **How to explain it:** "Our second AI branch is a Transformer—the same core tech behind ChatGPT. Instead of just looking at the zoomed-in dip, the Transformer looks at the entire 27-day observation. Its job is to find complex, long-range periodicities, like multi-planet systems, and differentiate them from random stellar variability."

### 3. The Stellar Expert (Physics MLP)
*   **The Metaphor:** The "Astrophysicist."
*   **How to explain it:** "AI without physics is blind. A 1% dip in light on a gigantic Sun-like star means something completely different than a 1% dip on a tiny Red Dwarf. Our third branch dynamically queries the NASA MAST API to pull 13 real physical traits of the star (Temperature, Radius, Gravity) so the AI can validate its guess against the laws of physics."

### 4. The Orchestrator (Confidence-Guided Gating)
*   **The Metaphor:** The "Boss."
*   **How to explain it:** "We don't just mash these three answers together. EvoNex uses an Adaptive Gating Network. Each of our three experts outputs a guess AND a confidence score. If the light curve is extremely noisy but the physics are clear, the Gating Network dynamically chooses to trust the Physics Expert more. It adapts on the fly to every single star."

---

## Part 3: The "Wow" Factors (To crush the competition)

If the judges ask what makes your project special, hit them with these two points:

1.  **Total Explainability:** "Because of our Gating Network, EvoNex is not a black box. During inference, it literally outputs its reasoning. It tells you, *'I believe this is an exoplanet, and I made this decision relying 45% on the CNN shape, 25% on the Transformer rhythm, and 30% on the Physics data.'* Judges and scientists demand explainability, and we built it in natively."
2.  **Ultra-Fast Engineering (HDF5 Caching):** "We built a custom PyTorch DataLoader that queries the NASA API, cleans the noise with Savitzky-Golay filters, folds the math using Box Least Squares, and then caches the massive tensor arrays into an HDF5 Database. This drops our data loading time from 5 seconds per star to 15 milliseconds—a 300x speed optimization."

---

## Part 4: Your Live Demo Script (Step-by-Step)
*When it's time to show the project working on your screen, follow these exact steps.*

### Step 1: Open the Visual Proofs
1.  Open your terminal in `C:\projects\AstroSage`.
2.  Run `.\venv\Scripts\Activate.ps1`
3.  Run `jupyter lab`
4.  Open `notebooks/01_EDA_and_Preprocessing.ipynb`.
    *   *Show them the top graph (messy gray dots).*
    *   *Scroll to the bottom graph (clean black dots).* Say: *"This is our mathematical preprocessing engine stripping out cosmic rays and telescope jitter in real-time."*
5.  Open `notebooks/03_BLS_Detection.ipynb`.
    *   *Scroll to the folded transit graph with the red highlight.* Say: *"This is our algorithm mathematically extracting the exact orbital period and transit duration before the AI even touches it."*

### Step 2: Run the Model Evaluation
1.  Open a new PowerShell terminal.
2.  Navigate to the folder and activate the environment again.
3.  Type: `python src/evaluate_evomoe.py` and hit Enter.
4.  As the text prints out on the screen, point out two things:
    *   **The F1-Score (0.945):** Show them the table proving that the EvoMoE is vastly superior to a standard CNN or Random Forest.
    *   **The Explainability Printout:** Highlight the text where the model outputs the exact percentages (e.g., *CNN Expert Contribution: 44.1%*). Say: *"And right there, you can see the model explaining its own decision-making process live."*
