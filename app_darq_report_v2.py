"""Gradio report-style app for the DARQ DaTSCAN quantification pipeline.

Place this file at the root of the repository, next to the `darq/`, `data/`,
and `scripts/` folders. Then run: `py app_darq_report.py`.

This app:
- uploads DaTSCAN, MRI/T1w and SynthSeg files;
- runs the existing `scripts/dat2mri.py` pipeline;
- shows a compact biomarker dashboard;
- shows clean SBR and symmetry tables;
- creates overlay images of the registered DaTSCAN over the MRI;
- creates a ZIP with all generated outputs;
- creates a simple PDF report for review.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import gradio as gr
import nibabel as nib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "gradio_runs"
APP_TITLE = "DARQ - DaTSCAN Quantification Report"


# -----------------------------------------------------------------------------
# File and execution helpers
# -----------------------------------------------------------------------------

def _find_pipeline_script() -> Path:
    """Return the expected pipeline script path."""
    candidates = [
        ROOT_DIR / "scripts" / "dat2mri.py",
        ROOT_DIR / "dat2mri.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No he encontrado el script del pipeline. Esperaba encontrar "
        "scripts/dat2mri.py o dat2mri.py en la raíz del proyecto."
    )


def _check_nifti_extension(path: Path) -> None:
    """Check that the uploaded file looks like a NIfTI image."""
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        raise ValueError(
            f"El archivo {path.name} no parece un NIfTI válido. "
            "Debe terminar en .nii o .nii.gz."
        )


def _copy_uploaded_file(uploaded_path: str | None, destination_dir: Path, name: str) -> Path:
    """Copy a Gradio-uploaded file to a stable run folder."""
    if uploaded_path is None:
        raise ValueError(f"Falta subir el archivo: {name}")

    source = Path(uploaded_path)
    if not source.exists():
        raise FileNotFoundError(f"No se ha encontrado el archivo subido: {source}")

    _check_nifti_extension(source)

    suffix = ".nii.gz" if source.name.endswith(".nii.gz") else source.suffix
    destination = destination_dir / f"{name}{suffix}"
    shutil.copy2(source, destination)
    return destination


def _read_first_tsv(output_dir: Path, pattern: str) -> pd.DataFrame:
    """Read the first TSV matching a pattern, or return an empty table."""
    matches = sorted(output_dir.glob(pattern))
    if not matches:
        return pd.DataFrame()
    return pd.read_csv(matches[0], sep="\t")


def _zip_output_folder(output_dir: Path) -> Path | None:
    """Create a zip file with all generated outputs."""
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return None
    zip_base = output_dir.parent / "results"
    return Path(shutil.make_archive(str(zip_base), "zip", output_dir))


# -----------------------------------------------------------------------------
# Biomarker tables
# -----------------------------------------------------------------------------

def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected columns to numeric when they exist."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _safe_div(numerator, denominator):
    """Safe scalar/Series division."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def _mean_sbr_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return only aggregate=mean rows from the raw SBR TSV."""
    if raw_df.empty:
        return pd.DataFrame()

    df = _to_numeric(raw_df, ["dat_str", "dat_cau", "dat_put", "dat_occ", "loss"])
    if "aggregate" in df.columns:
        df = df[df["aggregate"] == "mean"].copy()

    hemi_order = {"R": 0, "L": 1, "Both": 2}
    if "hemi" in df.columns:
        df["_hemi_order"] = df["hemi"].map(hemi_order).fillna(99)
        df = df.sort_values("_hemi_order")

    return df


def _get_hemi_value(df: pd.DataFrame, hemi: str, column: str) -> float:
    """Extract a single value from the mean SBR table."""
    if df.empty or "hemi" not in df.columns or column not in df.columns:
        return np.nan
    rows = df[df["hemi"] == hemi]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0][column])


def _sbr_value(df: pd.DataFrame, hemi: str, region_column: str) -> float:
    """Compute SBR = region / occipital - 1 for one hemisphere."""
    region = _get_hemi_value(df, hemi, region_column)
    occ = _get_hemi_value(df, hemi, "dat_occ")
    if not np.isfinite(region) or not np.isfinite(occ) or occ == 0:
        return np.nan
    return float(region / occ - 1)


def _asymmetry_index(right: float, left: float) -> float:
    """Compute AI = 100 * (Right - Left) / mean(Right, Left)."""
    if not np.isfinite(right) or not np.isfinite(left):
        return np.nan
    mean = (right + left) / 2
    if mean == 0:
        return np.nan
    return float(100 * (right - left) / mean)


def _clean_sbr_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw *_sbr.tsv table into a compact hemisphere table."""
    df = _mean_sbr_rows(raw_df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Hemisphere",
                "SBR striatum",
                "SBR caudate",
                "SBR putamen",
                "Occipital reference",
                "Registration loss",
            ]
        )

    rows = []
    for hemi in ["R", "L", "Both"]:
        rows.append(
            {
                "Hemisphere": hemi,
                "SBR striatum": _sbr_value(df, hemi, "dat_str"),
                "SBR caudate": _sbr_value(df, hemi, "dat_cau"),
                "SBR putamen": _sbr_value(df, hemi, "dat_put"),
                "Occipital reference": _get_hemi_value(df, hemi, "dat_occ"),
                "Registration loss": _get_hemi_value(df, hemi, "loss"),
            }
        )

    return pd.DataFrame(rows).round(4)


def _regional_sbr_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build a region x hemisphere table similar in spirit to volBrain reports."""
    df = _mean_sbr_rows(raw_df)
    if df.empty:
        return pd.DataFrame(columns=["Region", "Right SBR", "Left SBR", "Bilateral SBR", "Asymmetry index (%)"])

    mapping = {
        "Striatum": "dat_str",
        "Caudate": "dat_cau",
        "Putamen": "dat_put",
    }

    rows = []
    for region_name, column in mapping.items():
        r = _sbr_value(df, "R", column)
        l = _sbr_value(df, "L", column)
        both = _sbr_value(df, "Both", column)
        rows.append(
            {
                "Region": region_name,
                "Right SBR": r,
                "Left SBR": l,
                "Bilateral SBR": both,
                "Asymmetry index (%)": _asymmetry_index(r, l),
            }
        )
    return pd.DataFrame(rows).round(4)


def _clean_symmetry_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw *_symm.tsv table into a compact symmetry summary."""
    if raw_df.empty:
        return pd.DataFrame(columns=["Metric", "Hemisphere", "Caudate", "Putamen"])

    df = _to_numeric(raw_df, ["dat_cau", "dat_put"])
    if "aggregate" in df.columns:
        df = df[df["aggregate"] == "mean"].copy()

    metric_order = {"lncc": 0, "l2": 1}
    hemi_order = {"R": 0, "L": 1, "Both": 2}
    df["_metric_order"] = df.get("metric", pd.Series(index=df.index)).map(metric_order).fillna(99)
    df["_hemi_order"] = df.get("hemi", pd.Series(index=df.index)).map(hemi_order).fillna(99)
    df = df.sort_values(["_metric_order", "_hemi_order"])

    output = pd.DataFrame(
        {
            "Metric": df.get("metric", ""),
            "Hemisphere": df.get("hemi", ""),
            "Caudate": df.get("dat_cau", pd.NA),
            "Putamen": df.get("dat_put", pd.NA),
        }
    )
    return output.round(4)


def _build_metric_cards(raw_sbr: pd.DataFrame, completed: subprocess.CompletedProcess[str] | None) -> str:
    """Build HTML metric cards for the report header."""
    df = _mean_sbr_rows(raw_sbr)

    both_str = _sbr_value(df, "Both", "dat_str")
    both_cau = _sbr_value(df, "Both", "dat_cau")
    both_put = _sbr_value(df, "Both", "dat_put")
    r_put = _sbr_value(df, "R", "dat_put")
    l_put = _sbr_value(df, "L", "dat_put")
    ai_put = _asymmetry_index(r_put, l_put)
    loss = _get_hemi_value(df, "Both", "loss")

    if completed is None:
        qc = "C"
    elif completed.returncode == 0 and not raw_sbr.empty:
        qc = "A"
    elif completed.returncode == 0:
        qc = "B"
    else:
        qc = "C"

    def fmt(value: float, ndigits: int = 3) -> str:
        if not np.isfinite(value):
            return "—"
        return f"{value:.{ndigits}f}"

    items = [
        ("QC", qc, "A=ok · B=review · C=failed"),
        ("Striatum SBR", fmt(both_str), "Bilateral"),
        ("Caudate SBR", fmt(both_cau), "Bilateral"),
        ("Putamen SBR", fmt(both_put), "Bilateral"),
        ("Putamen AI", fmt(ai_put, 2) + "%" if np.isfinite(ai_put) else "—", "Right vs Left"),
        ("Reg. loss", fmt(loss, 4), "Final optimization loss"),
    ]

    cards = []
    for title, value, subtitle in items:
        cards.append(
            f"""
            <div style="border:1px solid #dddddd;border-radius:12px;padding:14px;margin:6px;background:#fafafa;min-width:145px;box-shadow:0 1px 3px rgba(0,0,0,0.12);">
                <div style="font-size:13px;color:#4b5563;">{title}</div>
                <div style="font-size:26px;font-weight:700;line-height:1.2;color:#111827;">{value}</div>
                <div style="font-size:12px;color:#6b7280;">{subtitle}</div>
            </div>
            """
        )
    return "<div style='display:flex;flex-wrap:wrap;'>" + "".join(cards) + "</div>"


def _build_interpretation_text(raw_sbr: pd.DataFrame) -> str:
    """Build a small explanation of what is being displayed."""
    if raw_sbr.empty:
        return "No se ha podido construir la interpretación porque no se ha encontrado el archivo SBR."

    return (
        "**Qué significa cada tarjeta**\n\n"
        "- **QC**: control de calidad automático de la ejecución. **A** = correcto, **B** = revisar, **C** = fallido.\n"
        "- **Striatum SBR**: captación específica media del estriado completo, usando la región occipital como referencia.\n"
        "- **Caudate SBR**: captación específica media del núcleo caudado.\n"
        "- **Putamen SBR**: captación específica media del putamen.\n"
        "- **Putamen AI**: índice de asimetría del putamen entre hemisferio derecho e izquierdo. Valores positivos indican Right > Left; valores negativos indican Left > Right.\n"
        "- **Reg. loss**: valor final de la función de pérdida del registro. Sirve como métrica técnica de optimización, no como biomarcador clínico.\n\n"
        "**Fórmulas mostradas en la app**\n\n"
        "- **SBR** = media de captación en la región / media de captación occipital - 1.\n"
        "- **Asymmetry index** = 100 × (Right - Left) / media(Right, Left).\n"
        "- La tabla completa original sigue guardada en el ZIP de resultados.\n\n"
        "Esta primera versión es una demo técnica de investigación; no debe interpretarse como informe clínico final."
    )


# -----------------------------------------------------------------------------
# Image visualization
# -----------------------------------------------------------------------------

def _find_resampled_dat(output_dir: Path) -> Path | None:
    """Find the registered/resampled DaT image generated by the pipeline."""
    candidates = sorted(output_dir.glob("*_desc-resampled_dat.nii.gz"))
    if candidates:
        return candidates[0]
    candidates = sorted(output_dir.glob("*_dat.nii.gz"))
    return candidates[0] if candidates else None


def _load_nifti_array(path: Path) -> np.ndarray:
    """Load a NIfTI image as a finite 3D float array."""
    arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _robust_normalize(arr: np.ndarray, lower: float = 1, upper: float = 99) -> np.ndarray:
    """Normalize an array to [0, 1] using robust percentiles."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [lower, upper])
    if hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _slice2d(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract and rotate one 2D slice for display."""
    index = int(np.clip(index, 0, arr.shape[axis] - 1))
    sl = np.take(arr, index, axis=axis)
    return np.rot90(sl)


def _resize_like(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D image for visualization only."""
    if image.shape == target_shape:
        return image
    try:
        from skimage.transform import resize
        return resize(image, target_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    except Exception:
        # Fallback: center-crop or pad if skimage resize fails.
        out = np.zeros(target_shape, dtype=np.float32)
        min0 = min(target_shape[0], image.shape[0])
        min1 = min(target_shape[1], image.shape[1])
        out[:min0, :min1] = image[:min0, :min1]
        return out


def _dat_center_indices(dat: np.ndarray) -> np.ndarray:
    """Choose display slices around the high-uptake DaT region."""
    if dat.size == 0 or np.max(dat) <= 0:
        return np.asarray(dat.shape) // 2
    threshold = np.percentile(dat[dat > 0], 95) if np.any(dat > 0) else np.percentile(dat, 95)
    coords = np.argwhere(dat >= threshold)
    if coords.size == 0:
        return np.asarray(dat.shape) // 2
    return np.round(np.median(coords, axis=0)).astype(int)


def _create_overlay_gallery(mri_path: Path, output_dir: Path, run_dir: Path) -> list[tuple[str, str]]:
    """Create axial/coronal/sagittal MRI-DaT overlay images for the Gradio gallery."""
    dat_path = _find_resampled_dat(output_dir)
    if dat_path is None:
        return []

    mri = _load_nifti_array(mri_path)
    dat = _load_nifti_array(dat_path)

    if mri.ndim != 3 or dat.ndim != 3:
        return []

    display_dir = run_dir / "display"
    display_dir.mkdir(exist_ok=True)

    dat_center = _dat_center_indices(dat)
    views = [
        ("Axial", 2),
        ("Coronal", 1),
        ("Sagittal", 0),
    ]

    gallery: list[tuple[str, str]] = []
    for view_name, axis in views:
        dat_idx = int(dat_center[axis])
        mri_idx = int(round(dat_idx / max(dat.shape[axis] - 1, 1) * max(mri.shape[axis] - 1, 1)))

        mri_slice = _robust_normalize(_slice2d(mri, axis, mri_idx))
        dat_slice = _robust_normalize(_slice2d(dat, axis, dat_idx), lower=5, upper=99.5)
        dat_slice = _resize_like(dat_slice, mri_slice.shape)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
        for ax in axes:
            ax.axis("off")

        axes[0].imshow(mri_slice, cmap="gray")
        axes[0].set_title("MRI / T1w", fontsize=10)

        axes[1].imshow(dat_slice, cmap="hot")
        axes[1].set_title("Registered DaTSCAN", fontsize=10)

        axes[2].imshow(mri_slice, cmap="gray")
        masked_dat = np.ma.masked_where(dat_slice <= 0.05, dat_slice)
        axes[2].imshow(masked_dat, cmap="hot", alpha=0.55)
        axes[2].set_title("Overlay", fontsize=10)

        fig.suptitle(f"{view_name} view", fontsize=12)
        fig.tight_layout()
        out_path = display_dir / f"overlay_{view_name.lower()}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        gallery.append((str(out_path), view_name))

    return gallery


# -----------------------------------------------------------------------------
# PDF report
# -----------------------------------------------------------------------------

def _add_dataframe_table(ax, df: pd.DataFrame, title: str) -> None:
    """Draw a compact dataframe table on a matplotlib axis."""
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    if df.empty:
        ax.text(0, 0.75, "No data available", fontsize=10)
        return

    display_df = df.copy().head(12)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)


def _create_pdf_report(
    run_dir: Path,
    subject_id: str,
    sbr_summary: pd.DataFrame,
    regional_table: pd.DataFrame,
    symm_summary: pd.DataFrame,
    overlay_gallery: list[tuple[str, str]],
    status_text: str,
) -> Path | None:
    """Create a simple PDF report using matplotlib."""
    pdf_path = run_dir / "DARQ_report.pdf"

    try:
        with PdfPages(pdf_path) as pdf:
            # Page 1: header + overlays + formula notes.
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.955, "DARQ report", fontsize=24, fontweight="bold")
            fig.text(0.06, 0.925, f"Subject: {subject_id or 'N/A'}", fontsize=11)
            fig.text(0.06, 0.905, f"Report date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=11)
            fig.text(0.06, 0.875, "DaTSCAN quantification with MRI/T1w and SynthSeg", fontsize=11)

            fig.text(
                0.06,
                0.835,
                "Main formulas: SBR = regional uptake / occipital uptake - 1; "
                "AI = 100 × (Right - Left) / mean(Right, Left).",
                fontsize=9,
            )

            if overlay_gallery:
                y_positions = [0.58, 0.36, 0.14]
                for (img_path, caption), y in zip(overlay_gallery[:3], y_positions):
                    image = plt.imread(img_path)
                    ax = fig.add_axes([0.06, y, 0.88, 0.18])
                    ax.imshow(image)
                    ax.axis("off")
                    ax.set_title(caption, fontsize=10, loc="left")
            else:
                fig.text(0.06, 0.70, "No overlay images were generated.", fontsize=11)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Page 2: tables.
            fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
            _add_dataframe_table(axes[0], regional_table, "Regional SBR summary")
            _add_dataframe_table(axes[1], sbr_summary, "Hemisphere SBR table")
            _add_dataframe_table(axes[2], symm_summary, "Symmetry summary")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Page 3: execution status.
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.95, "Execution log", fontsize=16, fontweight="bold")
            fig.text(0.06, 0.91, status_text[:5000], fontsize=7, family="monospace", va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        return pdf_path
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Main Gradio callback
# -----------------------------------------------------------------------------

def _build_status_message(
    completed: subprocess.CompletedProcess[str],
    output_dir: Path,
    sbr_raw: pd.DataFrame,
    symm_raw: pd.DataFrame,
) -> str:
    """Build a readable execution status message for the app."""
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode == 0:
        status = "Pipeline ejecutado correctamente.\n\n"
    else:
        status = "El pipeline ha terminado con error.\n\n"

    status += f"Carpeta de salida:\n{output_dir}\n\n"
    status += f"Filas SBR encontradas: {len(sbr_raw)}\n"
    status += f"Filas de simetría encontradas: {len(symm_raw)}\n\n"
    status += "Última salida de terminal:\n"
    status += stdout[-3000:] if stdout else "(sin salida estándar)"

    if stderr:
        status += "\n\nErrores / warnings:\n"
        status += stderr[-3000:]

    return status


def run_darq_report(
    dat_file: str | None,
    mri_file: str | None,
    seg_file: str | None,
    subject_id: str,
    optimizer: str,
    force: bool,
    use_cpu: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """Run the DARQ pipeline and return report-style outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_subject = (subject_id or "subject").strip().replace(" ", "_") or "subject"
    run_dir = RUNS_DIR / f"run_{timestamp}_{safe_subject}"
    input_dir = run_dir / "inputs"
    output_dir = run_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.05, desc="Preparing files")
        pipeline_script = _find_pipeline_script()
        dat_path = _copy_uploaded_file(dat_file, input_dir, "datscan")
        mri_path = _copy_uploaded_file(mri_file, input_dir, "T1w")
        seg_path = _copy_uploaded_file(seg_file, input_dir, "synthseg")

        command = [
            sys.executable,
            str(pipeline_script),
            "--dat",
            str(dat_path),
            "--mri",
            str(mri_path),
            "--seg",
            str(seg_path),
            "--o",
            str(output_dir),
            "--opt_str",
            optimizer,
        ]
        if force:
            command.append("--force")
        if use_cpu:
            command.append("--cpu")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR)

        progress(0.20, desc="Running DARQ pipeline")
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            env=env,
            text=True,
            capture_output=True,
            timeout=None,
        )

        progress(0.72, desc="Reading output tables")
        sbr_raw = _read_first_tsv(output_dir, "*_sbr.tsv")
        symm_raw = _read_first_tsv(output_dir, "*_symm.tsv")

        sbr_summary = _clean_sbr_table(sbr_raw)
        regional_table = _regional_sbr_table(sbr_raw)
        symm_summary = _clean_symmetry_table(symm_raw)
        cards_html = _build_metric_cards(sbr_raw, completed)
        interpretation = _build_interpretation_text(sbr_raw)
        status = _build_status_message(completed, output_dir, sbr_raw, symm_raw)

        progress(0.85, desc="Creating visual report")
        overlay_gallery = _create_overlay_gallery(mri_path, output_dir, run_dir)

        progress(0.93, desc="Creating downloads")
        zip_path = _zip_output_folder(output_dir)
        pdf_path = _create_pdf_report(
            run_dir=run_dir,
            subject_id=safe_subject,
            sbr_summary=sbr_summary,
            regional_table=regional_table,
            symm_summary=symm_summary,
            overlay_gallery=overlay_gallery,
            status_text=status,
        )

        progress(1.0, desc="Finished")
        return (
            cards_html,
            regional_table,
            sbr_summary,
            symm_summary,
            overlay_gallery,
            interpretation,
            status,
            str(zip_path) if zip_path else None,
            str(pdf_path) if pdf_path else None,
        )

    except Exception as exc:
        status = f"Error antes o durante la ejecución:\n{type(exc).__name__}: {exc}"
        empty = pd.DataFrame()
        return (
            _build_metric_cards(empty, None),
            empty,
            empty,
            empty,
            [],
            "No se ha podido generar la interpretación porque la ejecución ha fallado.",
            status,
            None,
            None,
        )


# -----------------------------------------------------------------------------
# Gradio interface
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
#main-title {text-align: center; margin-bottom: 0.2rem;}
#subtitle {text-align: center; color: #666; margin-bottom: 1.2rem;}
"""

with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
    gr.Markdown(f"# {APP_TITLE}", elem_id="main-title")
    gr.Markdown(
        "Sube los archivos del tutorial, ejecuta el pipeline y revisa un informe resumido con biomarcadores, imágenes y descargas.",
        elem_id="subtitle",
    )

    with gr.Tab("1. Inputs"):
        with gr.Row():
            dat_input = gr.File(label="DaTSCAN (.nii o .nii.gz)", type="filepath")
            mri_input = gr.File(label="MRI / T1w (.nii o .nii.gz)", type="filepath")
            seg_input = gr.File(label="SynthSeg (.nii o .nii.gz)", type="filepath")

        with gr.Row():
            subject_input = gr.Textbox(value="tutorial_subject", label="Subject ID")
            optimizer_input = gr.Dropdown(
                choices=["lbfgs", "adam", "sgd"],
                value="lbfgs",
                label="Optimizer",
            )
            force_input = gr.Checkbox(value=True, label="Force recompute")
            cpu_input = gr.Checkbox(value=True, label="Use CPU")

        run_button = gr.Button("Run DARQ report", variant="primary")

    with gr.Tab("2. Report summary"):
        cards_output = gr.HTML(label="Main biomarkers")
        interpretation_output = gr.Markdown(label="Interpretation notes")

    with gr.Tab("3. Images"):
        overlay_output = gr.Gallery(
            label="MRI / DaTSCAN / Overlay views",
            columns=1,
            height="auto",
            object_fit="contain",
        )

    with gr.Tab("4. Tables"):
        regional_output = gr.Dataframe(label="Regional SBR summary")
        sbr_output = gr.Dataframe(label="Hemisphere SBR summary")
        symm_output = gr.Dataframe(label="Symmetry summary")

    with gr.Tab("5. Execution log & downloads"):
        status_output = gr.Textbox(label="Status", lines=18)
        with gr.Row():
            zip_output = gr.File(label="Download complete output ZIP")
            pdf_output = gr.File(label="Download PDF report")

    run_button.click(
        fn=run_darq_report,
        inputs=[
            dat_input,
            mri_input,
            seg_input,
            subject_input,
            optimizer_input,
            force_input,
            cpu_input,
        ],
        outputs=[
            cards_output,
            regional_output,
            sbr_output,
            symm_output,
            overlay_output,
            interpretation_output,
            status_output,
            zip_output,
            pdf_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
