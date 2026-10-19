# Multimodal Medical Encoders: Classification & Retrieval

Two tasks on radiology figures from [OpenPMC](https://huggingface.co/datasets/vector-institute/open-pmc), using pre-trained medical vision-language models:

| Task | Model | Goal |
|---|---|---|
| Classification | [MedSigLIP](https://huggingface.co/google/medsiglip-448) (ViT-L/14, 448 px) | X-ray vs CT |
| Retrieval | [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (ViT-B/16 + PubMedBERT) | Image → matching caption |

Each task compares **zero-shot** against **fine-tuned**. Big Data course project from Nikita Chevokin's master's studies.

**Stack:** PyTorch · Hugging Face Transformers · open_clip · FAISS · scikit-learn · pandas/Parquet

---

## Pipeline

```
OpenPMC (streamed) → filter modality = radiology → label X-ray / CT from caption regex
   → 1,000 images (213 X-ray / 787 CT) → group split by PMC_ID (800 / 98 / 102)
   ├─ MedSigLIP: frozen backbone + MLP head (295,682 params) → X-ray vs CT
   └─ BiomedCLIP: vision frozen, text tower fine-tuned with memory bank → FAISS retrieval
```

## Key design decisions

- **Streaming data collection.** OpenPMC is too large to download, so records are streamed and filtered on the fly (radiology modality, usable caption, valid image).
- **Leak-free split.** `GroupShuffleSplit` on `PMC_ID`, so figures from one paper never straddle train and test.
- **Matching the pre-training preprocessing.** TF bilinear resize to 448×448 for MedSigLIP, as in the Big Vision pipeline.
- **Imbalance-aware evaluation.** 79% of the test set is CT, so results report macro-F1 and per-class recall next to accuracy. Training uses plain cross-entropy.
- **Memory bank for retrieval.** A 256-embedding circular buffer raises negatives per step from 7 to 263 at batch size 8, with no extra VRAM. Gradient accumulation (4 steps) gives an effective batch of 32.
- **Exact retrieval evaluation.** FAISS `IndexFlatIP` for image→text and text→image, reporting R@1/5/10 and MRR.

## Results (held-out test set, 102 images: 21 X-ray / 81 CT)

### Task 1: MedSigLIP, X-ray vs CT

| Model | Accuracy | Macro-F1 | X-ray recall | CT recall |
|---|---|---|---|---|
| Zero-shot | 0.882 | 0.798 | 0.57 | 0.96 |
| Fine-tuned | 0.902 | 0.845 | 0.71 | 0.95 |

Zero-shot is biased toward the majority class (CT). Fine-tuning mainly recovers X-ray sensitivity (15 of 21 X-rays found vs 12) at the cost of one CT image, a net gain of two correctly classified images, so the difference is within noise on this test set.

### Task 2: BiomedCLIP, caption retrieval

| Config | I2T R@1 | I2T R@5 | I2T R@10 | I2T MRR | T2I MRR |
|---|---|---|---|---|---|
| Zero-shot | 0.275 | 0.431 | 0.549 | 0.343 | 0.619 |
| Fine-tuned + memory bank | 0.284 | 0.490 | 0.598 | 0.377 | 0.608 |

Image→text improves modestly (MRR +0.034, R@5 +6 pts; R@1 differs by a single query out of 102); text→image is essentially unchanged. Image→text ranks against a 1,000-caption pool that includes captions seen in fine-tuning; text→image ranks against the 102 test images only.

## Limitations

- **Small test set.** Only 21 X-ray images: one misclassification moves macro-F1 by about 0.02, so differences between configurations are within noise.
- **Weak labels.** X-ray/CT labels come from caption regex, not from radiologists.
- **Single split, single seed.** No confidence intervals.

## Possible next steps

Cross-validation or bootstrap CIs, a larger sample from OpenPMC, manual label audit on a subset, more submodalities (MRI, ultrasound).

## Running it

Developed in Google Colab (GPU). `google/medsiglip-448` is gated: accept the licence on Hugging Face and set an `HF_TOKEN` secret.

```bash
pip install datasets transformers open_clip_torch faiss-cpu scikit-learn pandas pyarrow tensorflow
```

Open `notebook.ipynb` and run top to bottom. Data collection streams once and caches to `data/subset.parquet`.

## Repository

```
medical-multimodal-encoders/
├── README.md
├── notebook.ipynb
└── assets/          # exported result figures (confusion/bar charts, retrieval plot)
```
