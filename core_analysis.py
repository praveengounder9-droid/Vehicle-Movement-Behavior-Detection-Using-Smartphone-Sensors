"""
Core Analysis Module for Phone Motion and Driving Behavior Detection
Preserves the exact algorithm, formulas, thresholds, and classification rules
from the student's original project (Project.ipynb / L.D. College of Engineering).

Methodology:
1. Load accelerometer and gyroscope sensor files.
2. Remove duplicate rows from both datasets.
3. Select Milliseconds, X, Y, Z columns and rename to Acc_X, Acc_Y, Acc_Z and Gyro_X, Gyro_Y, Gyro_Z.
4. Merge on 'Milliseconds' using inner join.
5. Calculate magnitudes:
   - Acceleration_Magnitude = sqrt(Acc_X^2 + Acc_Y^2 + Acc_Z^2)
   - Gyroscope_Magnitude = sqrt(Gyro_X^2 + Gyro_Y^2 + Gyro_Z^2)
6. Calculate step-to-step changes:
   - Acceleration_Change = abs(diff(Acceleration_Magnitude))
   - Gyroscope_Change = abs(diff(Gyroscope_Magnitude))
7. Calculate statistical thresholds (mean + 2*std):
   - acc_threshold = Acceleration_Change.mean() + 2 * Acceleration_Change.std()
   - gyro_threshold = Gyroscope_Magnitude.mean() + 2 * Gyroscope_Magnitude.std()
8. Rule-based classification:
   - Default: 'Normal Movement'
   - If Gyroscope_Magnitude > gyro_threshold -> 'Turning'
   - Else if Acceleration_Change > acc_threshold and Gyroscope_Magnitude <= gyro_threshold
     -> 'Sudden Acceleration/Braking'
"""

import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Milliseconds", "X", "Y", "Z"]


def read_data_file(file_source: Union[str, io.BytesIO, io.StringIO, Any]) -> pd.DataFrame:
    """Read a CSV or Excel file from path or file-like object."""
    if isinstance(file_source, pd.DataFrame):
        return file_source.copy()

    # Check filename if available
    filename = getattr(file_source, "name", "")
    if isinstance(file_source, str):
        filename = file_source

    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_source)
    else:
        # Default to read_csv
        return pd.read_csv(file_source)


def validate_dataframe(df: pd.DataFrame, sensor_name: str = "Sensor") -> Tuple[bool, List[str]]:
    """Validate that required columns exist in the dataframe."""
    errors = []
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"{sensor_name} file is missing required columns: {', '.join(missing_cols)}")

    if len(df) == 0:
        errors.append(f"{sensor_name} dataset is empty.")

    return len(errors) == 0, errors


def process_sensor_datasets(
    acc_source: Union[str, io.BytesIO, Any],
    gyro_source: Union[str, io.BytesIO, Any]
) -> Dict[str, Any]:
    """
    Executes the exact data loading, cleaning, merging, feature calculation,
    threshold estimation, and event classification as defined in Project.ipynb.
    """
    # 1. Read files
    try:
        acc_raw = read_data_file(acc_source)
    except Exception as e:
        raise ValueError(f"Failed to read Accelerometer file: {str(e)}")

    try:
        gyro_raw = read_data_file(gyro_source)
    except Exception as e:
        raise ValueError(f"Failed to read Gyroscope file: {str(e)}")

    # 2. Validate columns
    acc_valid, acc_errors = validate_dataframe(acc_raw, "Accelerometer")
    if not acc_valid:
        raise ValueError("\n".join(acc_errors))

    gyro_valid, gyro_errors = validate_dataframe(gyro_raw, "Gyroscope")
    if not gyro_valid:
        raise ValueError("\n".join(gyro_errors))

    raw_acc_rows = len(acc_raw)
    raw_gyro_rows = len(gyro_raw)

    # 3. Drop duplicate rows (Preserves original code: accelerometer.drop_duplicates())
    acc_clean = acc_raw.drop_duplicates().copy()
    gyro_clean = gyro_raw.drop_duplicates().copy()

    acc_duplicates_removed = raw_acc_rows - len(acc_clean)
    gyro_duplicates_removed = raw_gyro_rows - len(gyro_clean)

    # 4. Prepare columns for merge
    # Keep Timestamp if present for display and time analysis
    acc_cols = ["Milliseconds", "X", "Y", "Z"]
    has_acc_timestamp = "Timestamp" in acc_clean.columns
    if has_acc_timestamp:
        acc_cols = ["Timestamp"] + acc_cols

    acc = acc_clean[acc_cols].copy()
    gyro = gyro_clean[["Milliseconds", "X", "Y", "Z"]].copy()

    acc.rename(columns={
        "X": "Acc_X",
        "Y": "Acc_Y",
        "Z": "Acc_Z"
    }, inplace=True)

    gyro.rename(columns={
        "X": "Gyro_X",
        "Y": "Gyro_Y",
        "Z": "Gyro_Z"
    }, inplace=True)

    acc_unique_ms = acc["Milliseconds"].nunique()
    gyro_unique_ms = gyro["Milliseconds"].nunique()

    # 5. Inner merge on Milliseconds (Exact original logic)
    merged_data = pd.merge(acc, gyro, on="Milliseconds", how="inner")
    merged_rows = len(merged_data)

    if merged_rows == 0:
        raise ValueError(
            "Inner merge resulted in 0 records. Ensure both Accelerometer and "
            "Gyroscope files share overlapping 'Milliseconds' values."
        )

    # Sort chronologically by Milliseconds to ensure diff() is consistent
    merged_data.sort_values(by="Milliseconds", inplace=True)
    merged_data.reset_index(drop=True, inplace=True)

    # 6. Calculate Magnitudes (Exact original formulas)
    # merged_data["Acceleration_Magnitude"] = np.sqrt(Acc_X^2 + Acc_Y^2 + Acc_Z^2)
    merged_data["Acceleration_Magnitude"] = np.sqrt(
        merged_data["Acc_X"] ** 2 + merged_data["Acc_Y"] ** 2 + merged_data["Acc_Z"] ** 2
    )

    # merged_data["Gyroscope_Magnitude"] = np.sqrt(Gyro_X^2 + Gyro_Y^2 + Gyro_Z^2)
    merged_data["Gyroscope_Magnitude"] = np.sqrt(
        merged_data["Gyro_X"] ** 2 + merged_data["Gyro_Y"] ** 2 + merged_data["Gyro_Z"] ** 2
    )

    # 7. Calculate Step Changes (diff().abs())
    merged_data["Acceleration_Change"] = merged_data["Acceleration_Magnitude"].diff().abs()
    merged_data["Gyroscope_Change"] = merged_data["Gyroscope_Magnitude"].diff().abs()

    # 8. Compute Statistical Thresholds (Exact original formulas: mean + 2 * std)
    acc_change_mean = float(merged_data["Acceleration_Change"].mean())
    acc_change_std = float(merged_data["Acceleration_Change"].std())
    acc_threshold = acc_change_mean + 2 * acc_change_std

    gyro_mag_mean = float(merged_data["Gyroscope_Magnitude"].mean())
    gyro_mag_std = float(merged_data["Gyroscope_Magnitude"].std())
    gyro_threshold = gyro_mag_mean + 2 * gyro_mag_std

    # 9. Classification (Exact original rule hierarchy)
    merged_data["Event"] = "Normal Movement"

    # Rule 1: Turning: high gyroscope magnitude
    merged_data.loc[
        merged_data["Gyroscope_Magnitude"] > gyro_threshold,
        "Event"
    ] = "Turning"

    # Rule 2: Sudden acceleration/braking
    # Applied only where rotation is not already classified as turning
    merged_data.loc[
        (merged_data["Acceleration_Change"] > acc_threshold) &
        (merged_data["Gyroscope_Magnitude"] <= gyro_threshold),
        "Event"
    ] = "Sudden Acceleration/Braking"

    # 10. Summary Statistics & Aggregations
    event_counts = merged_data["Event"].value_counts().to_dict()
    total_events = len(merged_data)
    event_percentages = {
        k: (v / total_events * 100.0) for k, v in event_counts.items()
    }

    # Ensure all three standard classes exist in dictionaries
    for cls in ["Normal Movement", "Sudden Acceleration/Braking", "Turning"]:
        if cls not in event_counts:
            event_counts[cls] = 0
            event_percentages[cls] = 0.0

    detected_anomalies = event_counts.get("Sudden Acceleration/Braking", 0) + event_counts.get("Turning", 0)
    most_common_class = max(event_counts.items(), key=lambda x: x[1])[0]

    mean_acc_mag = float(merged_data["Acceleration_Magnitude"].mean())
    mean_gyro_mag = float(merged_data["Gyroscope_Magnitude"].mean())

    stats_summary = {
        "raw_acc_rows": raw_acc_rows,
        "raw_gyro_rows": raw_gyro_rows,
        "acc_clean_rows": len(acc_clean),
        "gyro_clean_rows": len(gyro_clean),
        "acc_duplicates_removed": acc_duplicates_removed,
        "gyro_duplicates_removed": gyro_duplicates_removed,
        "acc_unique_ms": acc_unique_ms,
        "gyro_unique_ms": gyro_unique_ms,
        "merged_rows": merged_rows,
        "has_timestamp": has_acc_timestamp,
        "acc_threshold": acc_threshold,
        "gyro_threshold": gyro_threshold,
        "acc_change_mean": acc_change_mean,
        "acc_change_std": acc_change_std,
        "gyro_mag_mean": gyro_mag_mean,
        "gyro_mag_std": gyro_mag_std,
        "mean_acc_mag": mean_acc_mag,
        "mean_gyro_mag": mean_gyro_mag,
        "std_acc_mag": float(merged_data["Acceleration_Magnitude"].std()),
        "std_gyro_mag": float(merged_data["Gyroscope_Magnitude"].std()),
        "max_acc_mag": float(merged_data["Acceleration_Magnitude"].max()),
        "max_gyro_mag": float(merged_data["Gyroscope_Magnitude"].max()),
        "max_acc_change": float(merged_data["Acceleration_Change"].max()),
        "max_gyro_change": float(merged_data["Gyroscope_Change"].max()),
        "event_counts": event_counts,
        "event_percentages": event_percentages,
        "detected_anomalies": detected_anomalies,
        "most_common_class": most_common_class,
    }

    return {
        "dataframe": merged_data,
        "stats": stats_summary,
        "acc_raw_preview": acc_raw.head(5),
        "gyro_raw_preview": gyro_raw.head(5),
    }


def explain_single_observation(
    row: pd.Series,
    acc_threshold: float,
    gyro_threshold: float
) -> Dict[str, Any]:
    """
    Generates a dynamic, plain-English explanation of why an individual observation
    received its specific classification based on the project's exact decision rules.
    """
    ms = row.get("Milliseconds", "N/A")
    timestamp = row.get("Timestamp", None)
    acc_x = row.get("Acc_X", 0.0)
    acc_y = row.get("Acc_Y", 0.0)
    acc_z = row.get("Acc_Z", 0.0)
    gyro_x = row.get("Gyro_X", 0.0)
    gyro_y = row.get("Gyro_Y", 0.0)
    gyro_z = row.get("Gyro_Z", 0.0)

    acc_mag = row.get("Acceleration_Magnitude", 0.0)
    gyro_mag = row.get("Gyroscope_Magnitude", 0.0)
    acc_change = row.get("Acceleration_Change", np.nan)
    gyro_change = row.get("Gyroscope_Change", np.nan)
    event = row.get("Event", "Unknown")

    # Evaluate conditions
    is_gyro_high = (gyro_mag > gyro_threshold)
    is_acc_change_high = (not np.isnan(acc_change)) and (acc_change > acc_threshold)

    explanation_lines = []
    reasoning_summary = ""

    if event == "Turning":
        reasoning_summary = (
            f"Classified as **Turning** because the Gyroscope Magnitude ({gyro_mag:.3f} rad/s) "
            f"exceeds the dynamic threshold ({gyro_threshold:.3f} rad/s)."
        )
        explanation_lines.append(
            f"1. **Rule Evaluation**: Rule 1 checks if `Gyroscope_Magnitude > gyro_threshold`."
        )
        explanation_lines.append(
            f"2. **Observed Value**: Gyroscope Magnitude is **{gyro_mag:.3f} rad/s**, which is greater than threshold **{gyro_threshold:.3f} rad/s**."
        )
        if is_acc_change_high:
            explanation_lines.append(
                f"3. **Priority Note**: Acceleration Change is also high ({acc_change:.3f} m/s² > {acc_threshold:.3f} m/s²), "
                f"but according to the project's rule hierarchy, sharp rotational motion takes precedence and is categorized as Turning."
            )
        else:
            explanation_lines.append(
                f"3. **Acceleration Status**: Acceleration Change ({'N/A' if np.isnan(acc_change) else f'{acc_change:.3f}'} m/s²) "
                f"did not trigger Sudden Acceleration/Braking independently."
            )
        explanation_lines.append(
            "4. **Real-World Context**: Indicates significant angular velocity around the phone's axes. "
            "In driving, this typically corresponds to a sharp turn, curve, or lane change. "
            "*(Caveat: It could also reflect the phone tilting or rotating inside its mount).* "
        )

    elif event == "Sudden Acceleration/Braking":
        reasoning_summary = (
            f"Classified as **Sudden Acceleration/Braking** because the Acceleration Change ({acc_change:.3f} m/s²) "
            f"exceeds its threshold ({acc_threshold:.3f} m/s²), while Gyroscope Magnitude ({gyro_mag:.3f} rad/s) "
            f"remains at or below its threshold ({gyro_threshold:.3f} rad/s)."
        )
        explanation_lines.append(
            f"1. **Rule Evaluation**: Rule 2 checks if `(Acceleration_Change > acc_threshold) AND (Gyroscope_Magnitude <= gyro_threshold)`."
        )
        explanation_lines.append(
            f"2. **Observed Values**: Acceleration Change is **{acc_change:.3f} m/s²** (threshold: **{acc_threshold:.3f} m/s²**), "
            f"and Gyroscope Magnitude is **{gyro_mag:.3f} rad/s** (threshold: **{gyro_threshold:.3f} rad/s**)."
        )
        explanation_lines.append(
            "3. **Condition Satisfied**: Since rotational motion is below threshold, this event is identified as longitudinal motion change."
        )
        explanation_lines.append(
            "4. **Real-World Context**: Indicates a sharp sudden change in linear velocity. "
            "In driving, this typically corresponds to hard braking or rapid forward acceleration. "
            "*(Caveat: Smartphone magnitude difference detects sudden change without differentiating braking from acceleration or road bumps).* "
        )

    else:  # Normal Movement
        reasoning_summary = (
            f"Classified as **Normal Movement** because neither sensor reading exceeded its respective dynamic threshold."
        )
        explanation_lines.append(
            "1. **Rule Evaluation**: Evaluated against Rule 1 (Turning) and Rule 2 (Sudden Acceleration/Braking)."
        )
        explanation_lines.append(
            f"2. **Gyroscope Check**: Gyroscope Magnitude is **{gyro_mag:.3f} rad/s**, which is below the threshold of **{gyro_threshold:.3f} rad/s** (Turning not triggered)."
        )
        acc_str = "N/A (first record)" if np.isnan(acc_change) else f"{acc_change:.3f} m/s²"
        explanation_lines.append(
            f"3. **Acceleration Check**: Acceleration Change is **{acc_str}**, which is below or equal to threshold **{acc_threshold:.3f} m/s²** (Sudden Acceleration/Braking not triggered)."
        )
        explanation_lines.append(
            "4. **Real-World Context**: Sensor activity falls within normal statistical variance (within 2 standard deviations of the mean). "
            "Indicates cruising, steady driving, or standing still."
        )

    return {
        "event": event,
        "milliseconds": ms,
        "timestamp": timestamp,
        "reasoning_summary": reasoning_summary,
        "explanation_lines": explanation_lines,
        "is_gyro_high": is_gyro_high,
        "is_acc_change_high": is_acc_change_high,
        "acc_magnitude": acc_mag,
        "gyro_magnitude": gyro_mag,
        "acc_change": acc_change,
        "gyro_change": gyro_change,
        "acc_threshold": acc_threshold,
        "gyro_threshold": gyro_threshold,
        "raw_readings": {
            "Acc_X": acc_x, "Acc_Y": acc_y, "Acc_Z": acc_z,
            "Gyro_X": gyro_x, "Gyro_Y": gyro_y, "Gyro_Z": gyro_z
        }
    }


def answer_user_query(
    query: str,
    pipeline_result: Dict[str, Any],
    selected_row_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Answers user questions dynamically and deterministically using Python calculations
    directly against the currently loaded dataset. Never fabricates values.
    """
    q = query.strip().lower()
    stats = pipeline_result["stats"]
    df = pipeline_result["dataframe"]

    # Pre-calculate common properties
    total_records = stats["merged_rows"]
    acc_threshold = stats["acc_threshold"]
    gyro_threshold = stats["gyro_threshold"]
    event_counts = stats["event_counts"]
    event_pcts = stats["event_percentages"]

    # Helper for row lookup
    def get_selected_or_first_row():
        idx = selected_row_idx if (selected_row_idx is not None and 0 <= selected_row_idx < len(df)) else 0
        return idx, df.iloc[idx]

    # 1. Dataset content overview
    if any(k in q for k in ["what does my dataset contain", "dataset contain", "what data", "columns", "overview"]):
        answer = (
            f"**Dataset Summary:**\n\n"
            f"- **Raw Input Records**: {stats['raw_acc_rows']:,} Accelerometer readings and {stats['raw_gyro_rows']:,} Gyroscope readings.\n"
            f"- **Duplicates Removed**: {stats['acc_duplicates_removed']:,} duplicates from Accelerometer and {stats['gyro_duplicates_removed']:,} from Gyroscope.\n"
            f"- **Clean Merged Records**: **{total_records:,} synchronized observations** matched on `Milliseconds`.\n"
            f"- **Calculated Features**: 3-axis Accelerometer (`Acc_X`, `Acc_Y`, `Acc_Z`), 3-axis Gyroscope (`Gyro_X`, `Gyro_Y`, `Gyro_Z`), "
            f"`Acceleration_Magnitude`, `Gyroscope_Magnitude`, `Acceleration_Change`, and `Gyroscope_Change`.\n"
            f"- **Target Classification**: Rule-based `Event` column categorize into Normal Movement, Turning, or Sudden Acceleration/Braking."
        )
        return {"matched": True, "answer": answer, "category": "Dataset Overview"}

    # 2. Most common detected event
    if any(k in q for k in ["most common", "frequent", "highest count", "majority", "predominant"]):
        most_common = stats["most_common_class"]
        count = event_counts[most_common]
        pct = event_pcts[most_common]
        answer = (
            f"**Most Common Detected Event:**\n\n"
            f"The most common classification is **{most_common}** with **{count:,} observations** "
            f"({pct:.2f}% of all {total_records:,} merged records).\n\n"
            f"This is expected in driving behavior datasets because vehicles operate smoothly under normal conditions "
            f"most of the time, with sudden maneuvers and sharp turns occurring only intermittently."
        )
        return {"matched": True, "answer": answer, "category": "Distribution"}

    # 3. Peak/Highest Gyroscope Magnitude
    if ("gyroscope" in q or "gyro" in q) and any(k in q for k in ["highest", "maximum", "max", "peak", "highest?"]):
        max_idx = df["Gyroscope_Magnitude"].idxmax()
        max_row = df.loc[max_idx]
        ts_str = f" Timestamp: `{max_row['Timestamp']}`," if "Timestamp" in max_row else ""
        answer = (
            f"**Peak Gyroscope Reading:**\n\n"
            f"- **Maximum Gyroscope Magnitude**: **{max_row['Gyroscope_Magnitude']:.4f} rad/s**\n"
            f"- **Row Index**: **{max_idx}**\n"
            f"- **Timestamp / Milliseconds**: **{max_row['Milliseconds']} ms**{ts_str}\n"
            f"- **Event Classification**: **{max_row['Event']}**\n"
            f"- **Individual Axes**: `Gyro_X = {max_row['Gyro_X']:.4f}`, `Gyro_Y = {max_row['Gyro_Y']:.4f}`, `Gyro_Z = {max_row['Gyro_Z']:.4f}`\n\n"
            f"*(Note: Calculated Gyroscope Threshold is {gyro_threshold:.4f} rad/s, so this peak reading is {max_row['Gyroscope_Magnitude'] / gyro_threshold:.1f}x higher than threshold).*"
        )
        return {"matched": True, "answer": answer, "category": "Sensor Extrema"}

    # 4. Peak/Highest Acceleration Change / Magnitude
    if ("acceleration" in q or "acc" in q) and any(k in q for k in ["highest", "maximum", "max", "peak"]):
        max_idx = df["Acceleration_Magnitude"].idxmax()
        max_row = df.loc[max_idx]
        max_change_idx = df["Acceleration_Change"].idxmax()
        max_change_row = df.loc[max_change_idx]
        ts_str = f" Timestamp: `{max_change_row['Timestamp']}`," if "Timestamp" in max_change_row else ""
        answer = (
            f"**Peak Acceleration Readings:**\n\n"
            f"- **Maximum Acceleration Magnitude**: **{max_row['Acceleration_Magnitude']:.4f} m/s²** at Milliseconds `{max_row['Milliseconds']}` (Row {max_idx})\n"
            f"- **Maximum Acceleration Change ($\\Delta Acc$)**: **{max_change_row['Acceleration_Change']:.4f} m/s²** at Milliseconds `{max_change_row['Milliseconds']} ms`{ts_str} (Row {max_change_idx})\n"
            f"- **Classification at Peak Change**: **{max_change_row['Event']}**\n\n"
            f"*(Note: Calculated Acceleration Threshold is {acc_threshold:.4f} m/s²).*"
        )
        return {"matched": True, "answer": answer, "category": "Sensor Extrema"}

    # 5. Why was this observation classified as X?
    if "why" in q and any(k in q for k in ["classified", "categorized", "observation", "sudden", "turning", "normal", "braking"]):
        idx, row = get_selected_or_first_row()
        exp = explain_single_observation(row, acc_threshold, gyro_threshold)
        answer = (
            f"**Explanation for Selected Observation (Row {idx}, Milliseconds: {row['Milliseconds']}):**\n\n"
            f"- **Assigned Classification**: **{row['Event']}**\n"
            f"- **Core Reasoning**: {exp['reasoning_summary']}\n\n"
            + "\n".join(exp["explanation_lines"])
        )
        return {"matched": True, "answer": answer, "category": "Observation Explanation"}

    # 6. Normal Movement count & percentage
    if "normal" in q and any(k in q for k in ["movement", "count", "how many", "percentage", "percent", "ratio", "proportion", "classified"]):
        count = event_counts.get("Normal Movement", 0)
        pct = event_pcts.get("Normal Movement", 0.0)
        answer = (
            f"**Normal Movement Observations:**\n\n"
            f"- **Count**: **{count:,} records**\n"
            f"- **Percentage**: **{pct:.2f}%** of the dataset\n\n"
            f"These observations had acceleration changes $\\le {acc_threshold:.3f}$ m/s² and gyroscope magnitudes $\\le {gyro_threshold:.3f}$ rad/s."
        )
        return {"matched": True, "answer": answer, "category": "Event Statistics"}

    # 7. Turning percentage or count
    if "turning" in q and any(k in q for k in ["percent", "percentage", "count", "how many", "ratio", "observations", "classified"]):
        count = event_counts.get("Turning", 0)
        pct = event_pcts.get("Turning", 0.0)
        answer = (
            f"**Turning Events:**\n\n"
            f"- **Count**: **{count:,} observations**\n"
            f"- **Percentage**: **{pct:.2f}%** of the dataset\n\n"
            f"An observation is classified as **Turning** whenever its `Gyroscope_Magnitude` exceeds the statistical threshold of **{gyro_threshold:.3f} rad/s**."
        )
        return {"matched": True, "answer": answer, "category": "Event Statistics"}

    # 8. Sudden Acceleration / Braking
    if any(k in q for k in ["sudden acceleration", "braking", "sudden braking"]) and any(k in q for k in ["count", "how many", "percentage", "percent", "observations"]):
        count = event_counts.get("Sudden Acceleration/Braking", 0)
        pct = event_pcts.get("Sudden Acceleration/Braking", 0.0)
        answer = (
            f"**Sudden Acceleration / Braking Events:**\n\n"
            f"- **Count**: **{count:,} observations**\n"
            f"- **Percentage**: **{pct:.2f}%** of the dataset\n\n"
            f"Classified when `Acceleration_Change > {acc_threshold:.3f}` m/s² while `Gyroscope_Magnitude <= {gyro_threshold:.3f}` rad/s."
        )
        return {"matched": True, "answer": answer, "category": "Event Statistics"}

    # 9. Calculated Thresholds
    if any(k in q for k in ["threshold", "thresholds", "cutoff", "cutoffs", "limit"]):
        answer = (
            f"**Calculated Statistical Thresholds (Formula: Mean + 2 * Std):**\n\n"
            f"1. **Acceleration Change Threshold (`acc_threshold`)**: **{acc_threshold:.4f} m/s²**\n"
            f"   - Mean ($\\mu$): {stats['acc_change_mean']:.4f}\n"
            f"   - Std Dev ($\\sigma$): {stats['acc_change_std']:.4f}\n"
            f"   - Formula: `{stats['acc_change_mean']:.4f} + 2 * {stats['acc_change_std']:.4f} = {acc_threshold:.4f}`\n\n"
            f"2. **Gyroscope Magnitude Threshold (`gyro_threshold`)**: **{gyro_threshold:.4f} rad/s**\n"
            f"   - Mean ($\\mu$): {stats['gyro_mag_mean']:.4f}\n"
            f"   - Std Dev ($\\sigma$): {stats['gyro_mag_std']:.4f}\n"
            f"   - Formula: `{stats['gyro_mag_mean']:.4f} + 2 * {stats['gyro_mag_std']:.4f} = {gyro_threshold:.4f}`"
        )
        return {"matched": True, "answer": answer, "category": "Thresholds"}

    # 10. Summarize findings
    if any(k in q for k in ["summarize", "main findings", "summary", "key findings"]):
        answer = (
            f"**Executive Summary of Uploaded Sensor Data:**\n\n"
            f"1. **Clean Processing**: Started with {stats['raw_acc_rows']:,} accelerometer & {stats['raw_gyro_rows']:,} gyroscope rows; "
            f"after removing duplicates, **{total_records:,} synchronized records** were processed.\n"
            f"2. **Driving Stability**: **{event_pcts['Normal Movement']:.2f}%** ({event_counts['Normal Movement']:,} observations) "
            f"exhibit normal, smooth motion.\n"
            f"3. **Dynamic Maneuvers Detected**: A total of **{stats['detected_anomalies']} significant events** were detected:\n"
            f"   - **{event_counts['Sudden Acceleration/Braking']} Sudden Acceleration/Braking events** ({event_pcts['Sudden Acceleration/Braking']:.2f}%)\n"
            f"   - **{event_counts['Turning']} Turning events** ({event_pcts['Turning']:.2f}%)\n"
            f"4. **Sensor Baselines**: Average acceleration magnitude is **{stats['mean_acc_mag']:.2f} m/s²** "
            f"(near Earth's gravity $1g \\approx 9.8$ m/s²), and mean gyroscope magnitude is **{stats['mean_gyro_mag']:.2f} rad/s**.\n"
            f"5. **Dynamic Cutoffs**: Thresholds were established at +2 standard deviations: **{acc_threshold:.3f} m/s²** (acceleration change) "
            f"and **{gyro_threshold:.3f} rad/s** (gyroscope rotation)."
        )
        return {"matched": True, "answer": answer, "category": "Summary"}

    # 11. Limitations of results
    if any(k in q for k in ["limitation", "limitations", "drawback", "caveats", "accuracy"]):
        answer = (
            f"**Limitations of the Current Rule-Based Sensor Analysis:**\n\n"
            f"1. **Directional Ambiguity**: `Acceleration_Change` uses the absolute scalar difference `|diff(Magnitude)|`. "
            f"It cannot reliably distinguish whether the vehicle braked, accelerated forward, or hit a vertical road bump/pothole.\n"
            f"2. **Phone Rotation vs. Vehicle Turn**: A high `Gyroscope_Magnitude` indicates the phone rotated, but if the phone was picked up, "
            f"slid on a console, or tilted in a pocket, it triggers a 'Turning' event even if the vehicle stayed straight.\n"
            f"3. **Absence of Vehicle Orientation Alignment**: The phone axes are in device coordinates, not vehicle frame (Euler roll/pitch/yaw alignment was not applied).\n"
            f"4. **Heuristic 2-Sigma Threshold**: Using $\\mu + 2\\sigma$ assumes a quasi-normal distribution and dataset-wide adaptation; different driving trips will produce different threshold levels.\n"
            f"5. **File-Based vs Real-Time**: Processing is offline batch analysis on recorded sensor logs, not a real-time embedded safety trigger."
        )
        return {"matched": True, "answer": answer, "category": "Limitations"}

    # Fallback for unknown questions
    return {
        "matched": False,
        "answer": (
            f"I could not find a direct answer for: *'{query}'*.\n\n"
            f"**Try asking one of the suggested questions:**\n"
            f"- *What does my dataset contain?*\n"
            f"- *What is the most common detected event?*\n"
            f"- *How many observations were classified as Normal Movement?*\n"
            f"- *What percentage of observations were classified as Turning?*\n"
            f"- *What are the calculated thresholds?*\n"
            f"- *At which timestamp is the gyroscope magnitude highest?*\n"
            f"- *Why was this observation classified as Sudden Acceleration/Braking?*\n"
            f"- *Summarize the main findings from my uploaded data.*\n"
            f"- *What are the limitations of these results?*"
        ),
        "category": "Help"
    }


def generate_markdown_report(pipeline_result: Dict[str, Any]) -> str:
    """Generates a downloadable, academic-grade markdown report."""
    stats = pipeline_result["stats"]
    counts = stats["event_counts"]
    pcts = stats["event_percentages"]

    report = f"""# Mini Project Analysis Report
## Phone Motion and Driving Behavior Detection Using Accelerometer and Gyroscope

**Academic Context:**
- **Institution:** L.D. College of Engineering, Ahmedabad
- **Program:** B.E. Artificial Intelligence & Machine Learning (Semester 5)
- **Course:** Python Programming (BE05000241)
- **Author:** Gounder Praveen Murugesan (Enrollment: 240280152025)

---

### 1. Executive Summary
This report analyzes smartphone accelerometer and gyroscope data to detect vehicle motion events, including normal movement, sharp turns, and sudden acceleration or braking maneuvers. The system uses a rule-based statistical approach ($\mu + 2\sigma$ thresholds) applied to 3-axis sensor vector magnitudes.

### 2. Dataset Processing Metrics
| Metric | Accelerometer | Gyroscope | Combined / Merged |
| :--- | :--- | :--- | :--- |
| **Raw Input Records** | {stats['raw_acc_rows']:,} | {stats['raw_gyro_rows']:,} | - |
| **Duplicates Removed** | {stats['acc_duplicates_removed']:,} | {stats['gyro_duplicates_removed']:,} | - |
| **Clean Single-Sensor Records** | {stats['acc_clean_rows']:,} | {stats['gyro_clean_rows']:,} | - |
| **Merged Observations (`Milliseconds`)** | - | - | **{stats['merged_rows']:,}** |

### 3. Mathematical Formulas and Thresholds
1. **Sensor Vector Magnitudes:**
   - $\\text{{Acceleration Magnitude}} = \\sqrt{{Acc\\_X^2 + Acc\\_Y^2 + Acc\\_Z^2}}$
   - $\\text{{Gyroscope Magnitude}} = \\sqrt{{Gyro\\_X^2 + Gyro\\_Y^2 + Gyro\\_Z^2}}$
2. **Step Changes:**
   - $\\Delta \\text{{Acc}} = |\\text{{diff}}(\\text{{Acceleration Magnitude}})|$
   - $\\Delta \\text{{Gyro}} = |\\text{{diff}}(\\text{{Gyroscope Magnitude}})|$
3. **Dynamic Statistical Thresholds:**
   - **Acceleration Change Threshold:** $\\mu + 2\\sigma = {stats['acc_change_mean']:.4f} + 2({stats['acc_change_std']:.4f}) =$ **{stats['acc_threshold']:.4f} m/s²**
   - **Gyroscope Magnitude Threshold:** $\\mu + 2\\sigma = {stats['gyro_mag_mean']:.4f} + 2({stats['gyro_mag_std']:.4f}) =$ **{stats['gyro_threshold']:.4f} rad/s**

### 4. Event Classification Results
| Event Category | Count | Percentage | Classification Rule |
| :--- | :--- | :--- | :--- |
| **Normal Movement** | {counts['Normal Movement']:,} | {pcts['Normal Movement']:.2f}% | Default baseline ($\le$ both thresholds) |
| **Sudden Acceleration/Braking** | {counts['Sudden Acceleration/Braking']:,} | {pcts['Sudden Acceleration/Braking']:.2f}% | $\Delta \\text{{Acc}} > {stats['acc_threshold']:.3f}$ and $\\text{{Gyro}} \\le {stats['gyro_threshold']:.3f}$ |
| **Turning** | {counts['Turning']:,} | {pcts['Turning']:.2f}% | $\\text{{Gyroscope Magnitude}} > {stats['gyro_threshold']:.3f}$ |
| **Total Processed** | **{stats['merged_rows']:,}** | **100.00%** | - |

### 5. Key Findings
1. **Dominant State:** Normal driving accounts for {pcts['Normal Movement']:.2f}% of the observation window, confirming that typical driving is predominantly non-aggressive.
2. **Maneuver Detection:** The model isolated {stats['detected_anomalies']} anomalous events ({pcts['Sudden Acceleration/Braking']:.2f}% sudden speed changes and {pcts['Turning']:.2f}% sharp rotational changes).
3. **Gravity Baseline:** Mean acceleration magnitude is {stats['mean_acc_mag']:.2f} m/s², which aligns with normal Earth gravitational acceleration ($g \\approx 9.81$ m/s²).

### 6. Engineering Limitations & Caveats
- **Scalar Difference:** `Acceleration_Change` uses scalar difference rather than vector projection, so sudden braking and sudden forward acceleration cannot be reliably differentiated without vehicle orientation calibration.
- **Device vs. Vehicle Frame:** Gyroscope spikes detect phone reorientation. If the phone moves relative to the vehicle cradle, false turning classifications can occur.
- **Real-Time Context:** This offline analysis demonstrates feasibility for post-trip driver behavior profiling.
"""
    return report
