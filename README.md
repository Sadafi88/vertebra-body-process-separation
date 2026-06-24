# Vertebra Body and Posterior Process Separation

A classical image processing pipeline for separating vertebral bodies from posterior elements in labeled 3D spine segmentation masks.

This method does not require deep learning or training data.

---

## Features

- Processes 3D NIfTI masks (`.nii.gz`)
- Separates vertebral bodies and posterior processes
- PCA-based anatomical orientation estimation
- Gaussian Mixture Model (GMM) clustering
- Connected component refinement
- Connectivity enforcement for vertebral bodies
- Generates a combined labeled output mask

---

## Input

A labeled vertebra segmentation mask in NIfTI format:

```text
input_mask.nii.gz
```

Each vertebra should have its own label.

---

## Output

The output mask uses two label ranges:

```text
Vertebral body: original label
Posterior process: original label + 1000
```

Example:

```text
Vertebra 1 body      -> 1
Vertebra 1 process   -> 1001

Vertebra 2 body      -> 2
Vertebra 2 process   -> 1002
```

Output file:

```text
combined_vertebrae_mask_with_body_and_processes.nii.gz
```

---

## Method Overview

For each vertebra label, the pipeline:

1. Extracts vertebral voxel coordinates
2. Estimates anatomical orientation using PCA
3. Projects voxels into a lower-dimensional feature space
4. Applies Gaussian Mixture Model clustering
5. Identifies the vertebral body cluster
6. Refines posterior elements using connected component analysis
7. Enforces vertebral body connectivity
8. Generates a combined labeled output mask

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sadafi88/vertebra-body-process-separation.git
cd vertebra-body-process-separation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run:

```bash
python separate_vertebra_parts.py
```

Edit the input folder path in the script or modify it according to your dataset structure.

---

## Dependencies

- NumPy
- SciPy
- scikit-learn
- NiBabel
- SimpleITK

---

## Applications

- Spine CT segmentation post-processing
- Spine MRI segmentation post-processing
- Vertebral morphology analysis
- Dataset preparation for machine learning
- Anatomical structure separation

---

## Limitations

This is a classical image processing approach and may require adjustment for:

- Severe anatomical abnormalities
- Low-quality segmentations
- Highly fragmented vertebral masks

---

## Future Improvements

- Automatic parameter optimization
- Robust handling of pathological vertebrae
- Deep learning assisted refinement
- Quantitative evaluation metrics

---

## License

MIT License
