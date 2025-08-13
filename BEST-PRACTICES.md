# GrooVAE Humanize: Fine-tuning and Generation Strategies

This document distills practical, evidence-backed strategies to fine-tune and generate coherent MIDI with Google Magenta’s GrooVAE Humanize model, integrating key details from Magenta’s paper, official page, checkpoints docs, and demos.

## What the Humanize model is (and isn’t)

- The Humanize model takes a quantized drum score and generates expressive groove: microtiming and velocity, learned from real drummer performances rather than rule-based humanization.
- It is a MusicVAE variant trained with a Variational Information Bottleneck (VIB): the encoder is masked to see either score or groove, forcing the latent to model the missing component and smoothing the latent space for realistic outputs under variation.
- It was trained on the Groove MIDI Dataset (GMD), roughly 13–15 hours of tempo-aligned professional drumming with a metronome, enabling accurate modeling of deviations from the grid.

## Recommended pretrained configs

- groovae_2bar_humanize: a 2‑bar model intended to convert a quantized, constant‑velocity drum pattern into a humanized groove; widely referenced in Magenta’s docs and demos.
- Additional GrooVAE variants exist (e.g., tap2drum for “Drumify” and multi-bar autoencoders), but Humanize is the score→groove pathway you want for adding microtiming/velocity.

## Data preparation for fine-tuning

- Use expressive, tempo‑aligned drum performances that can be mapped to standard drum classes (e.g., GM subset used by Magenta).
- Maintain consistent bar length and grid with the target config (commonly 2 bars, 4/4, 16th grid).
- Prepare pairs that allow masking: a quantized version (score) and the expressive target (groove), mirroring the paper’s training setup.
- Keep instrument mapping and representation consistent with GrooVAE’s GrooveConverter/Drums mapping to minimize domain shift.

## Fine-tuning configuration tips

- Start from the groovae_2bar_humanize configuration to preserve task definition, bar length, representation, and masking objective.
- Keep the VIB objective and the same latent dimensionality/encoder–decoder architecture as the base; changes here can degrade the smooth latent space that supports realism and controllability.
- Use learning-rate warmup and modest training schedules to avoid catastrophic drift from the pretrained groove distributions, especially if your data style differs from GMD.
- Validate with both objective error (timing/velocity MAE) and listening tests; GrooVAE trades some exact reconstruction accuracy for musical realism and control.

## Input formatting for best generation results

- Feed the model clean, quantized, 2‑bar drum scores with constant velocities and standard drum pitches; Humanize adds the expressive layer.
- Align to a 16th grid (steps_per_quarter=4) and keep a steady tempo to match training assumptions from GMD recordings to a click.
- Stay within the model’s native length at inference (e.g., 2 bars) to avoid truncation or misalignment.

## Decoding and sampling settings (coherence-first)

- Use low sampling temperature (roughly 0.3–0.6) for conservative decoding that yields coherent, non‑chaotic grooves while maintaining a human feel.
- Decode at the configuration’s native sequence length (e.g., 2 bars for groovae_2bar_humanize) to preserve bar structure.
- The VIB prior ensures a smooth latent space and realistic outputs even for input scores that deviate from training distribution; conservative temperatures further stabilize outputs at scale.

## Generating “from scratch”

- Humanize is not unconditional: first create a simple, quantized 2‑bar score scaffold (e.g., four‑on‑the‑floor kick, snare on 2/4, 8th hats), then pass it through Humanize to render performance microtiming/velocity.
- To scale variety without losing coherence, vary the score minimally (kick placements, occasional ghost hats) and keep temperature modest; adjust temperature slightly upward if results feel too stiff, downward if they get busy or deviate from the scaffold.
- For longer pieces, humanize in 2‑bar chunks and stitch at bar boundaries to avoid phase drift and microtiming discontinuities.

## Practical fine-tuning workflow (high level)

- Convert your MIDI to NoteSequences and create training examples that permit masking (score vs groove).
- Train or fine‑tune using the groovae_2bar_humanize config with the same drum mapping, grid, and bar length as your inference setup.
- Monitor both quantitative timing/velocity metrics and qualitative listening; prioritize perceptual groove over exact token error if the goal is musical feel.
- Early-stop or use small learning rates if your target style is close to GMD; for markedly different styles, consider collecting targeted expressive recordings to avoid over-regularization toward GMD feel.

## Quality control and iteration

- If humanization adds unwanted fills or deviates from the prescribed score, reduce temperature and simplify the score’s density.
- If results lack feel, increase temperature slightly or introduce light accent patterns in the score (e.g., accented downbeats) to give the model more to respond to.
- Keep a fixed tempo per segment; variable tempo was not the focus of the original Humanize setup and can reduce stability.

## Key takeaways

- Separate concerns: use Humanize to add performance to a clean score, not to invent both score and groove.
- Match the model’s assumptions: 2‑bar length, steady tempo, 16th grid, standard drum mapping, quantized input.
- Favor conservative sampling: low temperature preserves coherence and leverages the VIB’s smooth latent space.
- Ensure representation and masking consistency across data, training, and inference to avoid domain shift and maintain the learned groove distributions.

## Minimal checklist

- Config/checkpoint: groovae_2bar_humanize (or your fine‑tuned variant).
- Input: 2‑bar quantized score, constant velocities, standard drum mapping, steady tempo.
- Decoding: native 2‑bar length; temperature ~0.3–0.6.
- Long form: process in 2‑bar chunks; stitch at bar boundaries.
- Fine‑tuning: expressive, tempo‑aligned drums similar to GMD; keep representation, masking, and grid consistent.

## Notes on the underlying research

- GrooVAE’s VIB setup masks either score or groove during training, encouraging a latent that robustly predicts the missing component and generalizes with realistic performance nuances.
- The Groove MIDI Dataset’s metronome‑aligned, professional performances are central to learning microtiming and velocity distributions that sound natural across common drum styles.
- Empirical findings in the paper show that while seq2seq baselines can minimize error more tightly, GrooVAE provides a controllable and perceptually convincing way to humanize patterns, particularly when used within its design constraints.

Sources
[1] magenta-demos/colab-notebooks/GrooVAE.ipynb at main https://github.com/magenta/magenta-demos/blob/main/colab-notebooks/GrooVAE.ipynb
[2] GrooVAE: Generating and Controlling Expressive Drum ... https://magenta.withgoogle.com/groovae
[3] magenta-js/music/checkpoints/README.md at master https://github.com/magenta/magenta-js/blob/master/music/checkpoints/README.md
[4] Humanizing MIDI files https://groups.google.com/a/tensorflow.org/g/magenta-discuss/c/OIlX7mk_mUQ
[5] Training a humanize GrooVAE model with own MIDI-data https://groups.google.com/a/tensorflow.org/g/magenta-discuss/c/cxnhtGtYhFE
[6] Hands-On Music Generation with Magenta: Explore https://booksrun.com/9781838824419-hands-on-music-generation-with-magenta-explore-the-role-of-deep-learning-in-music-generation-and-assisted-music-composition
[7] Google Magenta Plugins: GrooVAE https://www.youtube.com/watch?v=fyvaisDGoyU
[8] Magenta Studio - Ableton Live Plugin https://magenta.withgoogle.com/studio/
[9] Magenta.js Documentation https://magenta.github.io/magenta-js/
[10] Design, Development and Deployment of Real-Time Drum ... https://www.tesisenred.net/bitstream/handle/10803/693304/tbh.pdf?sequence=1&isAllowed=y
