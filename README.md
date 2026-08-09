# GDL
Avian-Inspired Task-Driven Pseudo-UV Synthesis and Multispectral Fusion for Camouflaged Wild Berry Detection# Avian-GDL

**Avian-Inspired Generation-Detection Loop for Task-Driven Pseudo-UV Synthesis and Multispectral Fusion**

Avian-GDL is an end-to-end, task-driven framework for **camouflaged wild berry detection** in complex agricultural and natural environments. The framework is motivated by avian ultraviolet vision: while berries can be difficult to distinguish from foliage in standard RGB images, extra-spectral modalities such as UV can reveal stronger target-background contrast.

The key idea is to **synthesize a discriminative Pseudo-UV modality from RGB images and directly optimize the synthesis process using downstream detection feedback**.

> **RGB → CA-UVGen → Pseudo-UV → Avian-MidDet → Detection**
>
> **Detection Loss → Perceptual Feedback → CA-UVGen**

This closed-loop design moves beyond conventional two-stage image translation, where a generator is optimized only for pixel-level reconstruction.

---

## ✨ Highlights

- 🐦 **Avian-inspired vision:** motivated by the ability of birds to exploit UV information for locating hidden food.
- 🌈 **Pseudo-UV synthesis:** generates extra-spectral information directly from RGB images without requiring a physical UV camera during inference.
- 🧠 **CA-UVGen:** Cross-Attentive UV Generator using a Vision Foundation Model (VFM) and dense cross-attention.
- 🔀 **Avian-MidDet:** mid-level multispectral fusion architecture combining RGB semantics and Pseudo-UV structural cues.
- 🔄 **Avian-GDL:** task-driven generation-detection feedback loop.
- 🎯 **Detection-aware generation:** the generator is optimized not only for visual reconstruction but also for downstream localization and detection.
- 🧹 **Task-optimized attention behavior:** Pseudo-UV can suppress irrelevant foliage/background noise and emphasize discriminative berry structures.
- 💰 **No physical UV sensor required during inference.**
- 🌐 **Cross-spectral generalization:** validated on both UV wild berry detection and NIR sweet pepper detection.

---

## 🧩 Framework Overview

The proposed framework contains three major components:

```text
                         RGB Image
                             │
                             ▼
                  ┌─────────────────────┐
                  │      CA-UVGen       │
                  │ Cross-Attentive UV  │
                  │      Generator      │
                  └──────────┬──────────┘
                             │
                             ▼
                      Pseudo-UV Image
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
             RGB Stream             Pseudo-UV Stream
                 │                       │
                 ▼                       ▼
            RGB CSP Stem           Pseudo-UV CSP Stem
                 │                       │
                 └───────────┬───────────┘
                             ▼
                    Cross-Modal CSP
                         Fusion
                             │
                             ▼
                      Deep Backbone
                             │
                             ▼
                       PAN / Neck
                             │
                             ▼
                    Detection Head
                             │
                             ▼
                       Predictions

          Detection Loss Ldet
                    │
                    │ Backpropagation
                    ▼
             CA-UVGen Parameters
```

The framework therefore establishes a differentiable generation-detection loop:

\[
RGB \rightarrow Pseudo\text{-}UV \rightarrow Detection
\rightarrow L_{det} \rightarrow Generator
\]

The detector's perceptual localization loss becomes a **task-driven reward signal** for the generator.

---

# 🧠 1. CA-UVGen

## Cross-Attentive UV Generator

CA-UVGen performs deterministic RGB-to-extra-spectral translation.

Given an RGB image:

\[
X_{rgb}\in\mathbb{R}^{H\times W\times3}
\]

the generator produces a single-channel Pseudo-UV representation:

\[
\hat{Y}_{spec}\in\mathbb{R}^{H\times W\times1}
\]

### Architecture

CA-UVGen consists of:

```text
RGB Input
   │
   ├───────────────────────┐
   │                       │
   ▼                       ▼
VFM Backbone          U-Net Generator
(Pre-trained/Frozen)  Encoder
   │                       │
   │                       ▼
   │                   Bottleneck
   │                       │
   └── Cross-Attention ────┤
                           ▼
                       Decoder
                           │
                           ▼
                      Pseudo-UV
```

The VFM extracts multi-scale semantic priors, while the convolutional generator maintains dense spatial correspondence.

The core formulation is:

\[
\hat{Y}_{spec}
=
G
\left(
X_{rgb},
Attn(Q_G,K_{vfm},V_{vfm})
\right)
\]

where:

- \(Q_G\) is derived from generator features;
- \(K_{vfm}\) and \(V_{vfm}\) are semantic keys and values from the VFM;
- `Attn` denotes dense cross-attention.

The VFM provides global/topological priors to guide the generator toward structurally aligned extra-spectral synthesis.

---

# 🔀 2. Avian-MidDet

## Mid-Level Multispectral Fusion

After Pseudo-UV synthesis, Avian-MidDet combines RGB and Pseudo-UV features at an intermediate semantic level.

Instead of direct pixel-level concatenation:

```text
Early Fusion
RGB ─┐
     ├── Concatenate → Detector
UV  ─┘
```

or independent late fusion:

```text
RGB ── Detector ──┐
                  ├── Prediction
UV  ── Detector ──┘
```

Avian-MidDet uses **mid-level fusion**:

```text
RGB Image
    │
    ▼
RGB CSP Stem
    │
    │
    ├──────────────┐
    │              │
    │        Cross-Modal CSP
    │             Fusion
    │              │
    │              ▼
    │        Deep Backbone
    │              │
    ▼              ▼
Pseudo-UV ──► Pseudo-UV CSP Stem
```

For spatial stage \(i\):

\[
f_{rgb}^{(i)}=\Phi_{rgb}^{(i)}(X_{rgb})
\]

\[
f_{spec}^{(i)}=\Phi_{spec}^{(i)}(\hat{Y}_{spec})
\]

The modality features are concatenated and passed through the Cross-Modal CSP fusion block:

\[
F_{agg}
=
\Psi_{fuse}
\left(
f_{rgb}^{(i)}
\oplus
f_{spec}^{(i)}
\right)
\]

The resulting representation is forwarded to the deep backbone, PAN/Neck, and detection head.

---

# 🔄 3. Avian-GDL

## Task-Driven Generation-Detection Loop

The central contribution of the framework is the coupling between generation and detection.

Conventional pseudo-multispectral pipelines typically follow:

```text
RGB → Generator → Synthetic UV
                     │
                     ▼
                 Detector
```

The generator is trained independently using reconstruction/adversarial objectives.

Avian-GDL instead establishes:

```text
             ┌───────────────────────┐
             │                       │
             ▼                       │
RGB → CA-UVGen → Pseudo-UV → Detector
                           │
                           ▼
                        Ldet
                           │
                           │ Gradient
                           ▼
                     CA-UVGen
```

This allows the generator to learn **what is useful for detection**, rather than simply reconstructing every physical characteristic of the UV sensor.

---

# ⚙️ 4. Two-Phase Optimization

Training is divided into two alternating phases.

## Phase 1 — Dual-Source Detector Optimization

The generator \(G\) is frozen.

The detector is optimized using both authentic physical pairs and generated synthetic pairs:

\[
L_{det}^{total}
=
L_{det}
\left(
D_{det}(X_{rgb},Y_{spec}),B_{gt}
\right)
+
L_{det}
\left(
D_{det}(X_{rgb},G(X_{rgb})),B_{gt}
\right)
\]

The detection loss includes bounding-box regression and classification errors, with the paper specifying GIoU and DFL components for localization.

This phase improves detector robustness against synthetic modality variation.

---

## Phase 2 — Perceptual Feedback-Guided Generator Optimization

The detector weights are frozen, but its computational graph remains active for gradient propagation.

The generator is optimized with:

\[
L_G^{total}
=
\left\|
G(X_{rgb})-Y_{spec}
\right\|_1
+
\lambda_{det}
L_{det}
\left(
D_{det}(X_{rgb},G(X_{rgb})),B_{gt}
\right)
\]

The task-driven feedback term is activated after **Epoch 30** in the reported experiments.

The generator therefore learns to balance:

```text
Pixel-level reconstruction
          +
Detection-oriented feature discrimination
```

---

# 🎯 5. Task-Realism Trade-off

An important finding of Avian-GDL is that the best modality for detection does **not necessarily need to look physically realistic**.

With only L1 reconstruction:

```text
Authentic UV
     ↓
Pixel-level similarity
     ↓
Lower FID
     ↓
More physical noise retained
```

With task-driven detection feedback:

```text
Authentic UV
     ↓
L1 Reconstruction
     +
Detection Feedback
     ↓
Task-optimized Pseudo-UV
     ↓
Background suppression
     +
Berry enhancement
     ↓
Improved detection
```

The paper reports that adding task-driven feedback increases FID from **65.00 to 85.73**, while improving:

- mAP@0.5: **0.644 → 0.737**
- Precision: **0.860 → 0.893**

This demonstrates a deliberate trade-off between physical visual fidelity and downstream task utility.

---

# 📊 6. Main Results

## Wild Berry Dataset — UV Domain

| Method | Modality | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|
| YOLOv8-s | RGB | 0.398 | 0.281 | 0.295 | 0.164 |
| YOLOv10-s | RGB | 0.493 | 0.233 | 0.268 | 0.160 |
| YOLOv11-s Baseline | RGB | 0.671 | 0.418 | 0.505 | 0.281 |
| YOLOv11-s Baseline | Physical UV | 0.443 | 0.368 | 0.387 | 0.218 |
| Avian-MidDet (Decoupled) | RGB + Real UV | 0.783 | 0.556 | 0.615 | 0.367 |
| Avian-GDL (Oracle) | RGB + Real UV | 0.862 | 0.637 | **0.740** | 0.487 |
| **Avian-GDL (Ours)** | **RGB + Pseudo-UV** | **0.893** | **0.638** | **0.737** | **0.476** |

The proposed method achieves:

- **Precision: 0.893**
- **Recall: 0.638**
- **mAP@0.5: 0.737**
- **mAP@0.5:0.95: 0.476**

The mAP@0.5 is extremely close to the joint-optimized physical UV oracle:

\[
0.737 \approx 0.740
\]

while the proposed Pseudo-UV system achieves higher precision:

\[
0.893 > 0.862
\]

---

# 🌶️ 7. Cross-Spectral Generalization

## Sweet Pepper Dataset — NIR Domain

The framework is also evaluated beyond UV using the NIR modality.

| Method | Modality | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|
| YOLOv8-s | RGB | 0.867 | 0.669 | 0.768 | 0.440 |
| YOLOv10-s | RGB | 0.830 | 0.682 | 0.784 | 0.471 |
| YOLOv11-s Baseline | RGB | 0.804 | 0.839 | 0.886 | 0.544 |
| YOLOv11-s Baseline | Physical NIR | 0.882 | 0.717 | 0.797 | 0.470 |
| Avian-MidDet (Decoupled) | RGB + Real NIR | 0.885 | 0.782 | 0.881 | 0.529 |
| Avian-GDL (Oracle) | RGB + Real NIR | 0.878 | 0.846 | **0.916** | **0.587** |
| **Avian-GDL (Ours)** | **RGB + Pseudo-NIR** | **0.862** | **0.819** | **0.915** | **0.581** |

The Pseudo-NIR model approaches the physical NIR oracle:

\[
mAP@0.5: 0.915 \approx 0.916
\]

This supports the cross-spectral applicability of the proposed task-driven generation paradigm.

---

# 🔬 8. Comparison with Pix2Next

The paper directly compares Avian-GDL with a decoupled Pix2Next-based generation-detection pipeline.

| Generator Strategy | Modalities | Joint Training | Precision | Recall | mAP@0.5 |
|---|---|:---:|---:|---:|---:|
| None (YOLOv11 RGB) | RGB | ✗ | 0.671 | 0.418 | 0.505 |
| Pix2Next + Fusion | RGB + Synthetic UV | ✗ | 0.773 | 0.561 | 0.621 |
| **CA-UVGen (Ours)** | **RGB + Pseudo-UV** | **✓** | **0.893** | **0.638** | **0.737** |

Compared with the decoupled Pix2Next pipeline:

\[
0.737-0.621=0.116
\]

Avian-GDL improves mAP@0.5 by **11.6 percentage points**.

The key difference is not simply the image-generation architecture, but the **task-driven feedback mechanism** that allows detection loss to optimize the generated modality.

---

# 📈 9. Ablation: Task-Driven Feedback

| L1 Pre-training | Task Feedback | FID ↓ | mAP@0.5 ↑ | Precision ↑ |
|---|---|---:|---:|---:|
| ✓ | ✗ | 65.00 | 0.644 | 0.860 |
| ✓ | ✓, starts Epoch 30 | 85.73 | **0.737** | **0.893** |

The experiment reveals an important phenomenon:

> **Lower visual reconstruction error does not necessarily mean better detection performance.**

Task-driven optimization intentionally allows the Pseudo-UV representation to deviate from authentic sensor appearance when doing so improves target discrimination.

---

# 📁 Repository Structure

A recommended implementation structure is:

```text
Avian-GDL/
│
├── README.md
│
├── models/
│   ├── ca_uvgen.py
│   ├── avian_middet.py
│   └── avian_gdl.py
│
├── modules/
│   └── cross_attention.py
│
├── configs/
│   ├── wildberry.yaml
│   └── sweetpepper.yaml
│
├── datasets/
│   ├── wildberry/
│   └── sweetpepper/
│
├── train_generator.py
├── train_detector.py
├── train_avian_gdl.py
├── inference.py
│
├── weights/
│
└── results/
```

---

# 🛠️ Installation

The reported experiments use:

- GPU: **NVIDIA GeForce RTX 4080**
- Image resolution: **384 × 384**
- Wild Berry training: **500 epochs**
- Sweet Pepper training: **300 epochs**
- Task-driven feedback activated at **Epoch 30**

Install the required deep-learning environment according to your implementation.

Example:

```bash
conda create -n avian-gdl python=3.10
conda activate avian-gdl

pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pillow
```

> Exact dependency versions are not fully specified in the paper. Adjust PyTorch/CUDA versions according to your GPU and implementation.

---

# 🚀 Training Pipeline

## Stage 1 — Generator Pre-training

First train CA-UVGen using paired RGB and authentic extra-spectral images:

```text
RGB + Authentic UV/NIR
          │
          ▼
       CA-UVGen
          │
          ▼
     Pseudo-UV/NIR
          │
          ▼
        L1 Loss
```

This establishes a stable initial cross-modal mapping.

---

## Stage 2 — Detector Optimization

Freeze the generator and optimize Avian-MidDet using:

```text
RGB + Real UV
       +
RGB + Pseudo-UV
       ↓
Avian-MidDet
       ↓
Detection Loss
```

---

## Stage 3 — Task-Driven Generator Optimization

Freeze the detector parameters but retain gradient propagation:

```text
RGB
 │
 ▼
CA-UVGen
 │
 ▼
Pseudo-UV
 │
 ▼
Avian-MidDet
 │
 ▼
Ldet
 │
 └────────────── Gradient ──────────────► CA-UVGen
```

Optimize:

\[
L_G
=
L_1+\lambda_{det}L_{det}
\]

The reported experiments activate the task-driven term from **Epoch 30**.

---

# 🧪 Inference

After training, only an RGB image is required:

```text
RGB Image
    │
    ├──────────────► RGB Branch
    │
    ▼
 CA-UVGen
    │
    ▼
Pseudo-UV
    │
    ▼
Pseudo-UV Branch
    │
    └──────────────┐
                   ▼
              Avian-MidDet
                   │
                   ▼
             Berry Detection
```

Therefore, the inference pipeline does not require a physical UV sensor.

---

# 🎯 Applications

Avian-GDL is designed for:

- Wild berry detection
- Robotic berry harvesting
- Agricultural robotics
- Camouflaged object detection
- RGB-to-UV synthesis
- RGB-to-NIR synthesis
- Multispectral object detection
- Cross-modal feature generation
- Precision agriculture
- Low-cost multispectral perception

The framework is particularly suitable for targets that are:

- visually camouflaged;
- small or partially occluded;
- surrounded by complex foliage;
- difficult to distinguish under RGB illumination;
- sensitive to false-positive detections.

---

# 💡 Key Insight

The central finding of Avian-GDL can be summarized as:

```text
Authentic Modality
       ≠
Optimal Detection Modality
```

A physical UV image contains real-world spectral information, but it can also contain irrelevant environmental noise.

The task-driven generator learns instead to produce:

```text
Pseudo-UV
    =
Discriminative Spectral Cues
    -
Irrelevant Background Noise
```

Consequently, Pseudo-UV behaves as an **implicit spatial attention mask** for the detector.

This explains why the proposed method achieves higher precision than the physical UV oracle:

\[
Precision_{PseudoUV}=0.893
\]

versus

\[
Precision_{PhysicalUV}=0.862
\]

---

# 📚 Citation

If you use Avian-GDL, CA-UVGen, or Avian-MidDet in your research, please cite:

```bibtex
@inproceedings{yang2026aviangdl,
  title     = {Avian-Inspired Task-Driven Pseudo-UV Synthesis and Multispectral Fusion for Camouflaged Wild Berry Detection},
  author    = {Yang, Hao and Yu, Junwei},
  affiliation = {Henan University of Technology, Zhengzhou, China}
}
```

> Please update the BibTeX entry with the final venue, year, pages, DOI, and official publication metadata after acceptance/publication.

---

# 📄 Paper

**Avian-Inspired Task-Driven Pseudo-UV Synthesis and Multispectral Fusion for Camouflaged Wild Berry Detection**

### Main contributions

1. **CA-UVGen**  
   A deterministic RGB-to-Pseudo-UV synthesis network based on VFM-guided cross-attention.

2. **Avian-MidDet**  
   A mid-level fusion detector that combines RGB semantic information with Pseudo-UV structural cues.

3. **Avian-GDL**  
   An end-to-end generation-detection feedback loop where detection loss directly optimizes pseudo-modality generation.

4. **Task-realism trade-off**  
   Demonstration that detection-optimized hallucinated modalities can outperform physically authentic modalities in precision.

---

# 🙏 Acknowledgements

This work builds upon research in:

- Vision Foundation Models
- Image-to-image translation
- Multispectral object detection
- Camouflaged object detection
- YOLO-based object detection
- Pix2Next
- Cycle-consistent learning

The paper specifically discusses Pix2Next, InternImage, YOLOv8/YOLOv10/YOLOv11, and prior multispectral detection methods as related foundations.

---

# 📜 License

Please add the license corresponding to the final research/code release policy before publishing this repository.

---

## ⭐ If Avian-GDL is useful for your research

Please consider citing the paper and giving the repository a ⭐.

[README_Avian-GDL.md](https://github.com/user-attachments/files/30871953/README_Avian-GDL.md)
