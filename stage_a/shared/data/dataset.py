import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import pandas as pd


def load_nii(path: str) -> np.ndarray:
    """Load a NIfTI file and return a float32 numpy array.

    Note:
      - nibabel returns arrays in (X, Y, Z) index order.
      - We keep it as numpy here; we will reorder axes for PyTorch later.
    """
    arr = nib.load(path).get_fdata().astype(np.float32)
    # Some pipelines may leave NaNs/inf; make it safe.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def hu_clip_and_scale(
    ct: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    out_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Clip CT (HU) and scale to a target range.

    Why this is useful:
      - CT volumes contain extreme outliers (metal, air, etc.).
      - Diffusion / autoencoder training is much more stable when inputs are
        in a bounded range (commonly [-1, 1]).

    Args:
        ct: CT volume in HU (float32), any shape.
        hu_min/hu_max: clipping window.
        out_range: target output range. Default (-1, 1).

    Returns:
        Scaled array with same shape as input.
    """
    ct = np.clip(ct, hu_min, hu_max)

    # Map [hu_min, hu_max] -> [0, 1]
    ct01 = (ct - hu_min) / (hu_max - hu_min + 1e-8)
    ct01 = np.clip(ct01, 0.0, 1.0)

    lo, hi = out_range
    ct_scaled = ct01 * (hi - lo) + lo
    return ct_scaled.astype(np.float32)


def apply_mask(ct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to CT.

    We treat mask > 0 as foreground.
    Outside the mask we set intensity to 0.0 (after scaling this is neutral-ish).
    """
    fg = (mask > 0).astype(np.float32)
    return (ct * fg).astype(np.float32)


def to_torch_3d(ct_xyz: np.ndarray) -> torch.Tensor:
    """Convert a (X, Y, Z) numpy volume to a PyTorch tensor (C, D, H, W).

    PyTorch 3D conv expects:
      (N, C, D, H, W)

    Our convention here:
      - Input nibabel array: (X, Y, Z)
      - Output tensor: (1, Z, Y, X) == (C, D, H, W)

    This keeps the 'slice axis' as D (depth).
    """
    # Ensure input is numpy array with float32 dtype
    ct_xyz = np.asarray(ct_xyz, dtype=np.float32)
    
    # Transpose: (X, Y, Z) -> (Z, Y, X)
    ct_transposed = np.transpose(ct_xyz, (2, 1, 0))
    
    # Ensure C-contiguous and float32 dtype
    if not ct_transposed.flags['C_CONTIGUOUS'] or ct_transposed.dtype != np.float32:
        ct_transposed = np.ascontiguousarray(ct_transposed, dtype=np.float32)
    
    # Add channel dimension: (Z, Y, X) -> (1, Z, Y, X)
    ct_zyx = ct_transposed[None, ...]
    
    # Convert to tensor - use torch.tensor instead of torch.from_numpy for better compatibility
    # torch.tensor creates a copy, which is safer
    return torch.tensor(ct_zyx, dtype=torch.float32)


def discover_patient_dirs(root_dir: str) -> List[str]:
    """Return a sorted list of patient subdirectories under root_dir."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    patient_dirs = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            patient_dirs.append(p)

    patient_dirs.sort()
    return patient_dirs


class CTROIVolumeDataset(Dataset):
    """Dataset for ROI CT volumes (NIfTI) stored one patient per folder.

    Expected folder layout (example):
      /Volumes/.../TTS_RM_V2/AD_12191953/
        AD_12191953_roi.nii.gz
        heart_ventricle_left_roi.nii.gz   (optional)

    This dataset is intentionally simple:
      - It loads CT ROI volumes.
      - Optionally loads an LV mask and applies it.
      - HU window + scaling is applied.
      - Output is a torch tensor (C, D, H, W) float32.
      - Optionally loads patient metadata (age, sex, label) from a CSV file.

    For diffusion autoencoder training:
      - You typically want fixed-size inputs (you already resampled to 128×128×64).
      - You typically want bounded intensities (default [-1, 1]).
    """

    def __init__(
        self,
        root_dir: str,
        ct_suffix: str = "_roi.nii.gz",
        mask_filename: Optional[str] = "heart_ventricle_left_roi.nii.gz",
        apply_lv_mask: bool = False,
        exclude_patient_ids: Optional[List[str]] = None,
        hu_window: Tuple[float, float] = (-1000.0, 400.0),
        out_range: Tuple[float, float] = (-1.0, 1.0),
        expected_shape_xyz: Optional[Tuple[int, int, int]] = (128, 128, 64),
        return_patient_id: bool = True,
        metadata_csv: Optional[str] = None,
        include_patient_csv: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.ct_suffix = ct_suffix
        self.mask_filename = mask_filename
        self.apply_lv_mask = apply_lv_mask
        self.exclude_patient_ids = set(exclude_patient_ids or [])
        self.hu_window = hu_window
        self.out_range = out_range
        self.expected_shape_xyz = expected_shape_xyz
        self.return_patient_id = return_patient_id

        # Load include_patient_csv if provided (for train/val/test splits)
        self.include_patient_ids = None
        if include_patient_csv is not None:
            include_df = pd.read_csv(include_patient_csv)
            # Support both 'patient_id' and 'ID' column names
            if "patient_id" in include_df.columns:
                self.include_patient_ids = set(include_df["patient_id"].astype(str).tolist())
            elif "ID" in include_df.columns:
                self.include_patient_ids = set(include_df["ID"].astype(str).tolist())
            else:
                # If no explicit column, try first column
                first_col = include_df.columns[0]
                self.include_patient_ids = set(include_df[first_col].astype(str).tolist())

        if metadata_csv is not None:
            meta_df = pd.read_csv(metadata_csv)
            required_cols = {"ID", "age", "sex", "case"}
            if not required_cols.issubset(meta_df.columns):
                missing = required_cols - set(meta_df.columns)
                raise ValueError(f"Metadata CSV missing required columns: {missing}")
            
            # Build meta_dict: support both roi_file-based and ID-based matching
            self.meta_dict = {}
            self.meta_by_id_prefix = {}  # For prefix matching (ID -> patient_id)
            
            for _, row in meta_df.iterrows():
                meta_info = {
                    "age": row["age"],
                    "sex": row["sex"],
                    "case": row["case"],
                }
                
                # If roi_file column exists, use it to extract patient_id
                if "roi_file" in row and pd.notna(row["roi_file"]):
                    roi_file = str(row["roi_file"])
                    # Extract patient_id from roi_file: "AAP_50415783_61F_roi.nii.gz" -> "AAP_50415783_61F"
                    if roi_file.endswith("_roi.nii.gz"):
                        patient_id = roi_file[:-len("_roi.nii.gz")]
                        self.meta_dict[patient_id] = meta_info
                
                # Also store by ID prefix for fallback matching
                id_str = str(row["ID"])
                self.meta_by_id_prefix[id_str] = meta_info
        else:
            self.meta_dict = {}
            self.meta_by_id_prefix = {}

        # Build index of samples
        patient_dirs = discover_patient_dirs(root_dir)

        items: List[Dict[str, str]] = []
        for pdir in patient_dirs:
            pid = os.path.basename(pdir)
            if pid in self.exclude_patient_ids:
                continue
            
            # Filter by include_patient_csv if provided
            if self.include_patient_ids is not None:
                # Try exact match first
                if pid not in self.include_patient_ids:
                    # Try prefix matching (in case CSV has ID prefix)
                    matched = False
                    for csv_id in self.include_patient_ids:
                        if pid.startswith(str(csv_id)) or str(csv_id).startswith(pid):
                            matched = True
                            break
                    if not matched:
                        continue

            # Match metadata: try exact match first, then prefix match
            has_metadata = False
            if self.meta_dict:
                if pid in self.meta_dict:
                    has_metadata = True
                elif self.meta_by_id_prefix:
                    # Try prefix matching: check if any ID prefix matches the start of patient_id
                    for id_prefix, _ in self.meta_by_id_prefix.items():
                        if pid.startswith(id_prefix):
                            # Update meta_dict with full patient_id for later retrieval
                            self.meta_dict[pid] = self.meta_by_id_prefix[id_prefix]
                            has_metadata = True
                            break
                # Skip if metadata required but not found
                if not has_metadata:
                    continue

            # Find CT file: {pid}{ct_suffix}
            ct_path = os.path.join(pdir, f"{pid}{self.ct_suffix}")
            if not os.path.isfile(ct_path):
                # Some datasets may name CT file differently; skip safely.
                continue

            mask_path = None
            if self.mask_filename is not None:
                candidate = os.path.join(pdir, self.mask_filename)
                if os.path.isfile(candidate):
                    mask_path = candidate

            items.append({
                "patient_id": pid,
                "ct_path": ct_path,
                "mask_path": mask_path or "",
            })

        if len(items) == 0:
            raise RuntimeError(
                f"No valid samples found under {root_dir}. "
                f"Expected CT file pattern: <patient_id>{ct_suffix}"
            )

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        pid = item["patient_id"]
        ct_path = item["ct_path"]
        mask_path = item["mask_path"]

        ct_xyz = load_nii(ct_path)  # (X, Y, Z)

        # Optional shape sanity check (helps catch wrong file / wrong stage output)
        if self.expected_shape_xyz is not None:
            if tuple(ct_xyz.shape) != tuple(self.expected_shape_xyz):
                raise ValueError(
                    f"[{pid}] Unexpected CT shape {ct_xyz.shape}. "
                    f"Expected {self.expected_shape_xyz}. File: {ct_path}"
                )

        # HU clip + scaling
        ct_xyz = hu_clip_and_scale(
            ct_xyz,
            hu_min=float(self.hu_window[0]),
            hu_max=float(self.hu_window[1]),
            out_range=self.out_range,
        )

        # Optional LV mask application
        if self.apply_lv_mask:
            if not mask_path:
                raise FileNotFoundError(
                    f"[{pid}] apply_lv_mask=True but mask file not found. "
                    f"Expected: {self.mask_filename} under {os.path.dirname(ct_path)}"
                )
            mask_xyz = load_nii(mask_path)

            # Basic alignment check (same grid expected for *_roi outputs)
            if mask_xyz.shape != ct_xyz.shape:
                raise ValueError(
                    f"[{pid}] CT and mask shapes differ: CT {ct_xyz.shape} vs mask {mask_xyz.shape}. "
                    f"CT: {ct_path} | mask: {mask_path}"
                )

            ct_xyz = apply_mask(ct_xyz, mask_xyz)

        x = to_torch_3d(ct_xyz).float()  # (1, Z, Y, X)

        if self.meta_dict:
            meta = self.meta_dict[pid]
            sex_str = str(meta["sex"]).upper().strip()
            if sex_str == 'F':
                sex_code = 0
            elif sex_str == 'M':
                sex_code = 1
            else:
                raise ValueError(
                    f"[{pid}] Invalid sex value: {meta['sex']}. Expected 'F' or 'M'."
                )
            age = float(meta["age"])
            case = int(meta["case"])

            if self.return_patient_id:
                return {
                    "x": x,
                    "y": case,
                    "meta": {
                        "age": age,
                        "sex": sex_code,
                    },
                    "patient_id": pid,
                    "ct_path": ct_path,
                }
            return {
                "x": x,
                "y": case,
                "meta": {
                    "age": age,
                    "sex": sex_code,
                },
            }
        else:
            if self.return_patient_id:
                return {
                    "x": x,
                    "patient_id": pid,
                    "ct_path": ct_path,
                }
            return x


def make_combined_dataset(
    tts_root: str,
    normal_root: str,
    exclude_patient_ids: Optional[List[str]] = None,
    apply_lv_mask: bool = False,
    hu_window: Tuple[float, float] = (-1000.0, 400.0),
    out_range: Tuple[float, float] = (-1.0, 1.0),
    tts_metadata_csv: Optional[str] = None,
    normal_metadata_csv: Optional[str] = None,
    expected_shape_xyz: Optional[Tuple[int, int, int]] = (128, 128, 64),
    return_patient_id: bool = True,
) -> Dataset:
    """Convenience function: concatenate TTS + Normal datasets.

    Args:
        tts_root: Root directory for TTS dataset.
        normal_root: Root directory for Normal dataset.
        exclude_patient_ids: List of patient IDs to exclude.
        apply_lv_mask: Whether to apply LV mask.
        hu_window: HU window for clipping.
        out_range: Output intensity range.
        tts_metadata_csv: Optional CSV file path for TTS metadata.
        normal_metadata_csv: Optional CSV file path for Normal metadata.
        expected_shape_xyz: Expected shape (X, Y, Z) for validation.
        return_patient_id: Whether to return patient_id in sample dict.

    Returns:
        torch.utils.data.ConcatDataset.
    """
    from torch.utils.data import ConcatDataset

    ds_tts = CTROIVolumeDataset(
        root_dir=tts_root,
        exclude_patient_ids=exclude_patient_ids,
        apply_lv_mask=apply_lv_mask,
        hu_window=hu_window,
        out_range=out_range,
        metadata_csv=tts_metadata_csv,
        expected_shape_xyz=expected_shape_xyz,
        return_patient_id=return_patient_id,
    )

    ds_normal = CTROIVolumeDataset(
        root_dir=normal_root,
        exclude_patient_ids=exclude_patient_ids,
        apply_lv_mask=apply_lv_mask,
        hu_window=hu_window,
        out_range=out_range,
        metadata_csv=normal_metadata_csv,
        expected_shape_xyz=expected_shape_xyz,
        return_patient_id=return_patient_id,
    )

    return ConcatDataset([ds_tts, ds_normal])