# UMUD Challenge notes

Competition: [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)

## Goal

Estimate from each B-mode ultrasound image:

- **PA** — pennation angle (deg): fascicle vs deep aponeurosis
- **FL** — fascicle length (mm): fascicle span between aponeuroses
- **MT** — muscle thickness (mm): perpendicular distance superficial ↔ deep aponeurosis

## Data layout (Kaggle)

| Path | Contents |
|------|----------|
| `apo_imgs` / `apo_masks` | ~1048 aponeurosis images + masks |
| `fasc_imgs` / `fasc_masks` | ~2761 fascicle images + masks |
| `test_images` | ~309 test images (includes 5-frame video snippets) |
| `sample_submission.csv` | Template (`image_id`, `pa_deg`, `fl_mm`, `mt_mm`; often `;` sep) |

Training labels are segmentation masks (Ritsche et al., 2024 UMB), not direct PA/FL/MT targets. Parameters are derived geometrically after segmentation.

## Evaluation

**UMUD Score** = normalized MAE over PA, FL, MT (lower better). Each MAE is scaled by an organizer-defined tolerance so units are comparable.

## Submission

```text
image_id,pa_deg,fl_mm,mt_mm
image_00001,15.2,78.5,19.1
```

Every test `image_id` exactly once.

## Prize eligibility

Provisional top-3 must also ship:

- OSI license (this repo: MIT)
- Public FAIR-ish docs + deps
- Reproducible inference (script / notebook / package)

## Allowed resources

Provided training data, public datasets indexed in UMUD, public pretrained models — document anything extra in `docs/method.md`.
