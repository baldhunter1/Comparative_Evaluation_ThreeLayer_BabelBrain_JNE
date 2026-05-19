'''

# CT SKULL-THICKNESS EXTRACTION ALONG EEG TRAJECTORIES

This script estimates CT-derived skull thickness along EEG-normal trajectories.

Important:
    Skull thickness is anatomical and not frequency-dependent.
    The 250 kHz CT folder is used only as the source folder for locating
    each subject's CT image and anatomical directory structure.

Main workflow:
    1. Search CT subject folders from the 250 kHz source directory.
    2. Load each subject's CT image and EEG-normal CSV.
    3. Exclude predefined EEG locations.
    4. Sample CT intensity along each inward EEG-normal trajectory.
    5. Estimate skull thickness from raw CT cortical-bone peaks.
    6. Save per-trajectory profiles, diagnostic figures, QC flags,
       and summary CSV files.

Outputs:
    - Per-trajectory intensity profiles
    - Per-trajectory diagnostic plots
    - Skull-thickness summary CSV
    - Thin-skull rows below 2.5 mm
    - Two-peak span rows
    - Flagged QC rows

Notes:
    CT skull threshold: 300 HU.
    Sampling window: +/- 40 mm around each EEG coordinate.
    Retained trajectories depend on the EEG-normal CSV and exclusion list.

Authors: 
    
    Moon Jeong, Dora Mackie

'''


from pathlib import Path
import re
import shutil

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import (
    map_coordinates,
    binary_closing,
    binary_opening,
    binary_fill_holes,
    label,
)


SEARCH_ROOTS = [
    {
        "modality": "CT",
        "frequency_khz": 250,
        "root": Path("/Volumes/Trans_2/transmission/data/CT/CT_f_250Hz"),
    },
]

EEG_NORMALS_DIR = Path("/Volumes/Trans_2/transmission/data/eeg_normals")

NUM_POINTS = 2400
RAY_HALF_LENGTH_MM = 40.0

CT_THRESHOLD = 300.0
MIN_COMPONENT_SIZE = 500
APPLY_MORPHOLOGY = True

CORTICAL_BONE_THRESHOLD_HU = 300.0
MIN_CORTICAL_SEGMENT_WIDTH_MM = 0.30
SCALP_TOLERANCE_BEHIND_ZERO_MM = 2.0

THIN_SKULL_THRESHOLD_MM = 2.5
TOO_THIN_THRESHOLD_MM = 1.0
TOO_THICK_THRESHOLD_MM = 20.0

OUT_SUBDIR = "skullthickness_CT_from_eeg_normals"


# =========================================================
# EXCLUDED TRAJECTORIES
# =========================================================

EXCLUDED_TRAJECTORIES = {
    "FP1", "FP2", "FPZ",
    "LPA", "NZ", "P9", "P10", "RPA",
    "AFZ", "AF8", "AF7", "AF4", "AF3",
    "IZ",
}


# =========================================================
# HELPERS
# =========================================================

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name).strip())


def normalize_electrode_name(name: str) -> str:
    return str(name).strip().upper()


def safe_rmtree(path: Path):
    if not path.exists():
        return

    def onerror(func, p, exc_info):
        err = exc_info[1]

        if isinstance(err, FileNotFoundError):
            return

        try:
            Path(p).chmod(0o777)
            func(p)
        except FileNotFoundError:
            return
        except Exception:
            raise

    shutil.rmtree(path, onerror=onerror)


def keep_large_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    lab, nlab = label(mask)

    if nlab == 0:
        return mask

    counts = np.bincount(lab.ravel())
    keep = counts >= min_size
    keep[0] = False

    return keep[lab]


def build_skull_mask(ct_data: np.ndarray, threshold: float = CT_THRESHOLD) -> np.ndarray:
    mask = ct_data > threshold

    if APPLY_MORPHOLOGY:
        mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
        mask = binary_opening(mask, structure=np.ones((3, 3, 3)))
        mask = binary_fill_holes(mask)
        mask = keep_large_components(mask, MIN_COMPONENT_SIZE)

    return mask.astype(np.uint8)


def find_contiguous_segments(binary_array: np.ndarray):
    idx = np.where(binary_array > 0)[0]

    if idx.size == 0:
        return []

    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]

    return list(zip(starts.tolist(), ends.tolist()))


def segment_to_info(distances, segment):
    s, e = segment
    start_mm = float(distances[s])
    end_mm = float(distances[e])
    thickness_mm = abs(end_mm - start_mm)

    return {
        "start_idx": int(s),
        "end_idx": int(e),
        "start_mm": min(start_mm, end_mm),
        "end_mm": max(start_mm, end_mm),
        "thickness_mm": float(thickness_mm),
        "mid_mm": float((start_mm + end_mm) / 2.0),
    }


def sample_along_normal(
    mask_data,
    raw_data,
    inv_affine,
    point_mm,
    normal_vec,
    num_points=NUM_POINTS,
    half_length_mm=RAY_HALF_LENGTH_MM,
):
    normal_vec = np.asarray(normal_vec, dtype=float)
    nrm = np.linalg.norm(normal_vec)

    if nrm == 0:
        raise ValueError("Zero-length normal vector")

    normal_vec = normal_vec / nrm
    point_mm = np.asarray(point_mm, dtype=float)

    distances = np.linspace(-half_length_mm, +half_length_mm, num_points)
    pts_mm = point_mm[None, :] + distances[:, None] * normal_vec[None, :]

    pts_vox = (inv_affine @ np.c_[pts_mm, np.ones(num_points)].T).T[:, :3]

    mask_profile = map_coordinates(
        mask_data.astype(float),
        pts_vox.T,
        order=0,
        mode="nearest",
    )

    raw_profile = map_coordinates(
        raw_data.astype(float),
        pts_vox.T,
        order=1,
        mode="nearest",
    )

    mask_profile = (mask_profile > 0.5).astype(np.uint8)
    segments = find_contiguous_segments(mask_profile)

    return distances, raw_profile, mask_profile, segments


def measure_skull_from_raw_ct_profile(
    distances: np.ndarray,
    raw_profile: np.ndarray,
):
    high_mask = (raw_profile >= CORTICAL_BONE_THRESHOLD_HU).astype(np.uint8)
    high_segments_raw = find_contiguous_segments(high_mask)

    high_info = []

    for seg in high_segments_raw:
        info = segment_to_info(distances, seg)

        if info["thickness_mm"] >= MIN_CORTICAL_SEGMENT_WIDTH_MM:
            high_info.append(info)

    if not high_info:
        span_mask = np.zeros_like(high_mask, dtype=np.uint8)
        diag = {
            "NumHighSegmentsRaw": len(high_segments_raw),
            "NumHighSegmentsUsed": 0,
            "DownstreamSegmentsIncluded": 0,
            "InterPeakGap_mm": np.nan,
            "MeasurementSource": "RAW_CT_TWO_PEAK_SIMPLE",
        }
        return None, high_mask, span_mask, "NO_HIGH_PEAK_FOUND", diag

    candidate_segments = [
        x for x in high_info
        if x["end_mm"] >= -SCALP_TOLERANCE_BEHIND_ZERO_MM
    ]

    candidate_segments = sorted(candidate_segments, key=lambda x: x["start_mm"])

    if len(candidate_segments) >= 2:
        first = candidate_segments[0]
        second = candidate_segments[1]

        start_idx = first["start_idx"]
        end_idx = second["end_idx"]

        start_mm = float(distances[start_idx])
        end_mm = float(distances[end_idx])
        thickness_mm = abs(end_mm - start_mm)

        inter_peak_gap_mm = second["start_mm"] - first["end_mm"]

        span_mask = np.zeros_like(high_mask, dtype=np.uint8)
        span_mask[start_idx:end_idx + 1] = 1

        chosen = {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_mm": min(start_mm, end_mm),
            "end_mm": max(start_mm, end_mm),
            "thickness_mm": float(thickness_mm),
        }

        diag = {
            "NumHighSegmentsRaw": len(high_segments_raw),
            "NumHighSegmentsUsed": len(high_info),
            "DownstreamSegmentsIncluded": 2,
            "InterPeakGap_mm": float(inter_peak_gap_mm),
            "MeasurementSource": "RAW_CT_TWO_PEAK_SIMPLE",
        }

        return chosen, high_mask, span_mask, "RAW_PROFILE_TWO_PEAK_SPAN", diag

    if len(candidate_segments) == 1:
        first = candidate_segments[0]
        selection_note = "ONLY_ONE_HIGH_PEAK_FOUND_AFTER_ZERO"
    else:
        first = sorted(
            high_info,
            key=lambda x: min(abs(x["start_mm"]), abs(x["end_mm"]), abs(x["mid_mm"]))
        )[0]
        selection_note = "FALLBACK_CLOSEST_HIGH_PEAK"

    span_mask = np.zeros_like(high_mask, dtype=np.uint8)
    span_mask[first["start_idx"]:first["end_idx"] + 1] = 1

    chosen = {
        "start_idx": int(first["start_idx"]),
        "end_idx": int(first["end_idx"]),
        "start_mm": float(first["start_mm"]),
        "end_mm": float(first["end_mm"]),
        "thickness_mm": float(first["thickness_mm"]),
    }

    diag = {
        "NumHighSegmentsRaw": len(high_segments_raw),
        "NumHighSegmentsUsed": len(high_info),
        "DownstreamSegmentsIncluded": 1,
        "InterPeakGap_mm": np.nan,
        "MeasurementSource": "RAW_CT_TWO_PEAK_SIMPLE",
    }

    return chosen, high_mask, span_mask, selection_note, diag


def qc_flag(chosen_seg, selection_note, diag):
    if chosen_seg is None:
        return "FAIL_NO_SKULL"

    t = chosen_seg["thickness_mm"]

    flags = []

    if selection_note == "RAW_PROFILE_TWO_PEAK_SPAN":
        flags.append("TWO_PEAK_SPAN")

    if selection_note == "ONLY_ONE_HIGH_PEAK_FOUND_AFTER_ZERO":
        flags.append("ONLY_ONE_HIGH_PEAK_FOUND")

    if selection_note == "FALLBACK_CLOSEST_HIGH_PEAK":
        flags.append("FALLBACK_CLOSEST_HIGH_PEAK")

    if t < TOO_THIN_THRESHOLD_MM:
        flags.append("TOO_THIN")

    if t < THIN_SKULL_THRESHOLD_MM:
        flags.append("LT_2P5MM")

    if t > TOO_THICK_THRESHOLD_MM:
        flags.append("TOO_THICK")

    if diag.get("DownstreamSegmentsIncluded", 0) == 2:
        flags.append("CORTICAL_TABLES_COMBINED")

    if not flags:
        return "OK"

    return ";".join(flags)


def find_ct_image(anat_dir: Path):
    candidates = sorted(
        p for p in anat_dir.glob("*_CT.nii.gz")
        if not p.name.startswith("._")
    )

    return candidates[0] if candidates else None


def get_subject_from_anat_dir(anat_dir: Path):
    return anat_dir.parent.name


def find_eeg_normals_csv(subject: str):
    subject_num = subject.replace("sub-", "")
    p = EEG_NORMALS_DIR / f"m2m_{subject_num}_eeg_normals.csv"

    return p if p.exists() else None


def load_eeg_normals(csv_path: Path):
    df = pd.read_csv(csv_path)

    required = ["Name", "R", "A", "S", "Nx", "Ny", "Nz"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    df["ElectrodeUpper"] = df["Name"].apply(normalize_electrode_name)

    before = len(df)
    df = df[~df["ElectrodeUpper"].isin(EXCLUDED_TRAJECTORIES)].copy()
    after = len(df)

    print(f"  excluded trajectories removed: {before - after}")
    print(f"  retained trajectories: {after}")

    return df


# =========================================================
# MAIN PER FOLDER
# =========================================================

def run_folder(anat_dir: Path, modality: str, frequency_khz: int):
    subj = get_subject_from_anat_dir(anat_dir)
    prefix = f"{subj}_{modality}_{frequency_khz}kHz_"

    image_path = find_ct_image(anat_dir)

    if image_path is None:
        print(f"SKIP (no CT image): {anat_dir}")
        return pd.DataFrame()

    eeg_csv = find_eeg_normals_csv(subj)

    if eeg_csv is None:
        print(f"SKIP (no EEG normals CSV): {subj}")
        return pd.DataFrame()

    img = nib.load(str(image_path))
    raw_data = img.get_fdata()
    inv_affine = np.linalg.inv(img.affine)

    skull_mask_3d = build_skull_mask(raw_data, threshold=CT_THRESHOLD)

    eeg_df = load_eeg_normals(eeg_csv)

    out_dir = anat_dir / OUT_SUBDIR

    if out_dir.exists():
        print(f"  removing old output folder: {out_dir}")
        safe_rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRUN CT skull thickness from 250 kHz folder  {subj}")
    print(f"  image   : {image_path.name}")
    print(f"  eeg csv : {eeg_csv.name}")
    print(f"  points  : {len(eeg_df)}")
    print(f"  out     : {out_dir}")

    records = []

    for _, row in eeg_df.iterrows():
        elec_name_original = str(row["Name"]).strip()
        elec_name = sanitize(elec_name_original)
        elec_upper = normalize_electrode_name(elec_name_original)

        if elec_upper in EXCLUDED_TRAJECTORIES:
            continue

        point = np.array([row["R"], row["A"], row["S"]], dtype=float)
        normal = np.array([row["Nx"], row["Ny"], row["Nz"]], dtype=float)

        try:
            distances, raw_profile, skull_mask_3d_profile, skull_mask_3d_segments = sample_along_normal(
                skull_mask_3d,
                raw_data,
                inv_affine,
                point,
                normal,
            )

            chosen_seg, high_mask_raw, skull_span_mask, selection_note, diag = measure_skull_from_raw_ct_profile(
                distances,
                raw_profile,
            )

            flag = qc_flag(chosen_seg, selection_note, diag)

            if chosen_seg is None:
                thickness = np.nan
                outer_mm = np.nan
                inner_mm = np.nan
                thin_skull = False
            else:
                outer_mm = chosen_seg["start_mm"]
                inner_mm = chosen_seg["end_mm"]
                thickness = chosen_seg["thickness_mm"]
                thin_skull = thickness < THIN_SKULL_THRESHOLD_MM

            pd.DataFrame({
                "Distance_mm": distances,
                "RawIntensity": raw_profile,
                "DiagnosticSkullMask3D": skull_mask_3d_profile,
                "CorticalBoneMaskRawProfile": high_mask_raw,
                "FinalSkullSpanMask": skull_span_mask,
            }).to_csv(
                out_dir / f"{prefix}profile_{elec_name}.csv",
                index=False,
            )

            fig, axes = plt.subplots(
                4,
                1,
                sharex=True,
                figsize=(7, 9),
                gridspec_kw={"height_ratios": [3, 1, 1, 1]},
            )

            axes[0].plot(distances, raw_profile)
            axes[0].axvline(0, color="black", linestyle=":")
            axes[0].axhline(CORTICAL_BONE_THRESHOLD_HU, linestyle=":", linewidth=1)

            if chosen_seg is not None:
                axes[0].axvline(outer_mm, color="green", linestyle="--")
                axes[0].axvline(inner_mm, color="red", linestyle="--")

            axes[0].set_ylabel("CT intensity")
            axes[0].set_title(
                f"{elec_name} | thickness = {thickness:.2f} mm | {flag}"
                if chosen_seg is not None
                else f"{elec_name} | thickness = NaN | {flag}"
            )
            axes[0].grid(True)

            axes[1].plot(distances, skull_mask_3d_profile)
            axes[1].axvline(0, color="black", linestyle=":")

            if chosen_seg is not None:
                axes[1].axvline(outer_mm, color="green", linestyle="--")
                axes[1].axvline(inner_mm, color="red", linestyle="--")

            axes[1].set_ylabel("3D mask")
            axes[1].set_ylim(-0.1, 1.1)
            axes[1].grid(True)

            axes[2].plot(distances, high_mask_raw)
            axes[2].axvline(0, color="black", linestyle=":")

            if chosen_seg is not None:
                axes[2].axvline(outer_mm, color="green", linestyle="--")
                axes[2].axvline(inner_mm, color="red", linestyle="--")

            axes[2].set_ylabel("Raw HU > 300")
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].grid(True)

            axes[3].imshow(
                skull_span_mask[np.newaxis, :],
                cmap="Greys",
                aspect="auto",
                origin="lower",
                extent=[distances[0], distances[-1], 0, 1],
            )
            axes[3].axvline(0, color="black", linestyle=":")

            if chosen_seg is not None:
                axes[3].axvline(outer_mm, color="green", linestyle="--")
                axes[3].axvline(inner_mm, color="red", linestyle="--")

            axes[3].set_xlabel("Distance along EEG normal (mm)")
            axes[3].set_yticks([])

            plt.tight_layout()

            fig.savefig(
                out_dir / f"{prefix}traj_{elec_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

            records.append({
                "Subject": subj,
                "Modality": modality,
                "SourceFolderFrequency_kHz": frequency_khz,
                "Electrode": elec_name_original,
                "ElectrodeUpper": elec_upper,

                "SkullThickness_mm": thickness,
                "SkullThickness_lt_2p5mm": thin_skull,
                "ThinSkullThreshold_mm": THIN_SKULL_THRESHOLD_MM,
                "TooThinThreshold_mm": TOO_THIN_THRESHOLD_MM,
                "TooThickThreshold_mm": TOO_THICK_THRESHOLD_MM,

                "OuterBoundary_mm": outer_mm,
                "InnerBoundary_mm": inner_mm,

                "SelectionNote": selection_note,
                "QCFlag": flag,

                "MeasurementSource": diag["MeasurementSource"],
                "Num3DMaskSegments": len(skull_mask_3d_segments),
                "NumHighSegmentsRaw": diag["NumHighSegmentsRaw"],
                "NumHighSegmentsUsed": diag["NumHighSegmentsUsed"],
                "DownstreamSegmentsIncluded": diag["DownstreamSegmentsIncluded"],
                "InterPeakGap_mm": diag["InterPeakGap_mm"],

                "CorticalBoneThreshold_HU": CORTICAL_BONE_THRESHOLD_HU,
                "MinCorticalSegmentWidth_mm": MIN_CORTICAL_SEGMENT_WIDTH_MM,
                "ScalpToleranceBehindZero_mm": SCALP_TOLERANCE_BEHIND_ZERO_MM,

                "ImageUsed": image_path.name,
                "EEGNormalsCSV": eeg_csv.name,
                "Diagnostic3DMaskThreshold_HU": CT_THRESHOLD,
                "HemiRayLength_mm": RAY_HALF_LENGTH_MM,

                "R": row["R"],
                "A": row["A"],
                "S": row["S"],
                "Nx": row["Nx"],
                "Ny": row["Ny"],
                "Nz": row["Nz"],
            })

        except Exception as e:
            records.append({
                "Subject": subj,
                "Modality": modality,
                "SourceFolderFrequency_kHz": frequency_khz,
                "Electrode": elec_name_original,
                "ElectrodeUpper": elec_upper,

                "SkullThickness_mm": np.nan,
                "SkullThickness_lt_2p5mm": False,
                "ThinSkullThreshold_mm": THIN_SKULL_THRESHOLD_MM,
                "TooThinThreshold_mm": TOO_THIN_THRESHOLD_MM,
                "TooThickThreshold_mm": TOO_THICK_THRESHOLD_MM,

                "OuterBoundary_mm": np.nan,
                "InnerBoundary_mm": np.nan,

                "SelectionNote": "EXCEPTION",
                "QCFlag": f"ERROR: {e}",

                "MeasurementSource": "RAW_CT_TWO_PEAK_SIMPLE",
                "Num3DMaskSegments": np.nan,
                "NumHighSegmentsRaw": np.nan,
                "NumHighSegmentsUsed": np.nan,
                "DownstreamSegmentsIncluded": np.nan,
                "InterPeakGap_mm": np.nan,

                "CorticalBoneThreshold_HU": CORTICAL_BONE_THRESHOLD_HU,
                "MinCorticalSegmentWidth_mm": MIN_CORTICAL_SEGMENT_WIDTH_MM,
                "ScalpToleranceBehindZero_mm": SCALP_TOLERANCE_BEHIND_ZERO_MM,

                "ImageUsed": image_path.name,
                "EEGNormalsCSV": eeg_csv.name,
                "Diagnostic3DMaskThreshold_HU": CT_THRESHOLD,
                "HemiRayLength_mm": RAY_HALF_LENGTH_MM,

                "R": row["R"],
                "A": row["A"],
                "S": row["S"],
                "Nx": row["Nx"],
                "Ny": row["Ny"],
                "Nz": row["Nz"],
            })

    df_out = pd.DataFrame(records)

    out_csv = out_dir / f"{prefix}skull_thickness_summary.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv}")

    thin_df = df_out[
        df_out["SkullThickness_mm"].notna()
        & (df_out["SkullThickness_mm"] < THIN_SKULL_THRESHOLD_MM)
    ].copy()

    out_thin_csv = out_dir / f"{prefix}skull_thickness_less_than_2p5mm.csv"
    thin_df.to_csv(out_thin_csv, index=False)
    print(f"  saved thin-skull CSV: {out_thin_csv}")
    print(f"  thin-skull rows < {THIN_SKULL_THRESHOLD_MM} mm: {len(thin_df)}")

    two_peak_df = df_out[
        df_out["SelectionNote"].astype(str).str.contains("TWO_PEAK", na=False)
    ].copy()

    out_two_peak_csv = out_dir / f"{prefix}two_peak_span_rows.csv"
    two_peak_df.to_csv(out_two_peak_csv, index=False)
    print(f"  saved two-peak CSV: {out_two_peak_csv}")
    print(f"  two-peak rows: {len(two_peak_df)}")

    flagged_df = df_out[df_out["QCFlag"] != "OK"].copy()
    out_flagged_csv = out_dir / f"{prefix}flagged_QC_rows.csv"
    flagged_df.to_csv(out_flagged_csv, index=False)
    print(f"  saved flagged QC CSV: {out_flagged_csv}")
    print(f"  flagged QC rows: {len(flagged_df)}")

    return df_out


# =========================================================
# DRIVER
# =========================================================

def main():
    all_records = []

    for entry in SEARCH_ROOTS:
        modality = entry["modality"]
        frequency_khz = entry["frequency_khz"]
        root = entry["root"]

        if not root.exists():
            print(f"WARNING: root does not exist: {root}")
            continue

        for anat_dir in sorted(root.glob("sub-*/anat")):
            df = run_folder(anat_dir, modality, frequency_khz)

            if not df.empty:
                all_records.append(df)

    if all_records:
        df_all = pd.concat(all_records, ignore_index=True)

        master_out_dir = Path("/Volumes/Trans_2/transmission/data/CT")
        master_out_dir.mkdir(parents=True, exist_ok=True)

        out_master = master_out_dir / "CT_skull_thickness_summary_250kHz_from_eeg_normals.csv"
        df_all.to_csv(out_master, index=False)

        print(f"\nSaved master summary: {out_master}")

        thin_master = df_all[
            df_all["SkullThickness_mm"].notna()
            & (df_all["SkullThickness_mm"] < THIN_SKULL_THRESHOLD_MM)
        ].copy()

        out_thin_master = master_out_dir / "CT_skull_thickness_less_than_2p5mm_250kHz_from_eeg_normals.csv"
        thin_master.to_csv(out_thin_master, index=False)

        print(f"Saved master thin-skull CSV: {out_thin_master}")
        print(f"Total rows < {THIN_SKULL_THRESHOLD_MM} mm: {len(thin_master)}")

        two_peak_master = df_all[
            df_all["SelectionNote"].astype(str).str.contains("TWO_PEAK", na=False)
        ].copy()

        out_two_peak_master = master_out_dir / "CT_two_peak_span_rows_250kHz_from_eeg_normals.csv"
        two_peak_master.to_csv(out_two_peak_master, index=False)

        print(f"Saved master two-peak CSV: {out_two_peak_master}")
        print(f"Total two-peak rows: {len(two_peak_master)}")

        flagged_master = df_all[df_all["QCFlag"] != "OK"].copy()

        out_flagged_master = master_out_dir / "CT_flagged_QC_rows_250kHz_from_eeg_normals.csv"
        flagged_master.to_csv(out_flagged_master, index=False)

        print(f"Saved master flagged QC CSV: {out_flagged_master}")
        print(f"Total flagged QC rows: {len(flagged_master)}")

        excluded_check = df_all[
            df_all["ElectrodeUpper"].isin(EXCLUDED_TRAJECTORIES)
        ]

        if len(excluded_check) > 0:
            print("\nWARNING: excluded trajectories still present in output:")
            print(
                excluded_check[
                    ["Subject", "SourceFolderFrequency_kHz", "Electrode", "ElectrodeUpper"]
                ].drop_duplicates().to_string(index=False)
            )
        else:
            print("\nQC PASSED: no excluded trajectories in output.")

        print("\nQC summary:")
        print(f"  Total rows: {len(df_all)}")
        print(f"  Thin skull rows < {THIN_SKULL_THRESHOLD_MM} mm: {len(thin_master)}")
        print(f"  Two-peak rows: {len(two_peak_master)}")
        print(f"  Flagged QC rows: {len(flagged_master)}")

        if "QCFlag" in df_all.columns:
            print("\nQCFlag counts:")
            print(df_all["QCFlag"].value_counts(dropna=False).to_string())

        if "SelectionNote" in df_all.columns:
            print("\nSelectionNote counts:")
            print(df_all["SelectionNote"].value_counts(dropna=False).to_string())

    print("\nDONE.")


if __name__ == "__main__":
    main()