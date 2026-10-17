# Phone Motion and Driving Behavior Detection Dashboard

*Student: Gounder Praveen Murugesan | Enrollment: 240280152025*

---

## 📌 Overview
This application transforms the rule-based smartphone sensor analysis from `Project.ipynb` into a modern, interactive web dashboard. It processes 3-axis Accelerometer and Gyroscope datasets recorded during vehicle motion to classify driving behaviors into:
1. **Normal Movement** (Smooth cruising / baseline driving)
2. **Sudden Acceleration / Braking** (Longitudinal speed changes)
3. **Turning** (Lateral rotation / cornering)

---

## 🔬 Core Methodology & Mathematical Formulas

1. **Duplicate Purging**: Identifies and removes duplicate timestamp logs from both `Accelerometer.csv` and `Gyroscope.csv`.
2. **Sensor Synchronization**: Merges sensor streams on elapsed `Milliseconds` via inner join.
3. **Sensor Vector Magnitudes**:
   Acceleration Magnitude = $\sqrt{Acc\_X^2 + Acc\_Y^2 + Acc\_Z^2}$
   Gyroscope Magnitude = $\sqrt{Gyro\_X^2 + Gyro\_Y^2 + Gyro\_Z^2}$
4. **Step-to-Step Changes**:
   $$\Delta \text{Acc} = |\text{diff}(\text{Acceleration Magnitude})|$$
   $$\Delta \text{Gyro} = |\text{diff}(\text{Gyroscope Magnitude})|$$
5. **Dynamic Statistical Thresholds ($\mu + 2\sigma$)**:
   acc_threshold $= \text{mean}(\Delta \text{Acc}) + 2 \times \text{std}(\Delta \text{Acc}) \approx 2.7958\text{ m/s}^2$
   gyro_threshold $= \text{mean}(\text{Gyro Magnitude}) + 2 \times \text{std}(\text{Gyro Magnitude}) \approx 1.2733\text{ rad/s}$
6. **Hierarchical Rule Classification**:
   - **Rule 1 (Turning)**:  If Gyroscope Magnitude > gyro threshold => Turning
   - **Rule 2 (Sudden Acc/Braking)**: If Delta Acc > acc threshold AND Gyroscope Magnitude <= gyro threshold => Sudden Acceleration/Braking
   - **Rule 3 (Normal Movement)**: Default baseline when neither threshold is breached.

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify with Test Suite
```bash
python test_pipeline.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Then open your web browser at: **`http://localhost:8501`**

---

## 📂 Project Structure
- [`core_analysis.py`](file:///d:/Users/kerat/Downloads/Project/core_analysis.py): Pure Python data processing module preserving original algorithms, formulas, threshold calculations, and deterministic Q&A engine.
- [`app.py`](file:///d:/Users/kerat/Downloads/Project/app.py): Complete interactive Streamlit web dashboard with Plotly charts, dark-mode glassmorphic styling, observation inspector, and export features.
- [`test_pipeline.py`](file:///d:/Users/kerat/Downloads/Project/test_pipeline.py): Automated test suite ensuring exact numerical equivalence with the notebook.
- [`Project.ipynb`](file:///d:/Users/kerat/Downloads/Project/Project.ipynb): Original research notebook (preserved intact).
- [`Accelerometer.csv`](file:///d:/Users/kerat/Downloads/Project/Accelerometer.csv): Recorded smartphone accelerometer data.
- [`Gyroscope.csv`](file:///d:/Users/kerat/Downloads/Project/Gyroscope.csv): Recorded smartphone gyroscope data.
- [`requirements.txt`](file:///d:/Users/kerat/Downloads/Project/requirements.txt): Python package dependencies.
