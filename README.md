# AI-Driven Prediction Model and Decision Support System

This repository contains the **Prediction Models and Decision Support System** for forecasting and managing the economic impact of cross-border dam water releases in Nigeria.

---

## Key Features

- **Flood Severity Prediction:** Classifies flood severity levels based on real-time inputs.
- **Economic Impact Estimation:** Predicts financial losses caused by floods.
- **Water Level Forecasting:** Projects water levels using hydrological and meteorological data.

---

## Technologies Used

- **FastAPI:** For deploying RESTful API services.
- **Streamlit:** Interactive frontend for user interaction.
- **Random Forest Models:** Core algorithm for classification and regression tasks.

---

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/matthewtech-o/LagdoDam-PaidX
cd project_directory
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the FastAPI Server
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8002
```
### 4. Start the Streamlit Application
```bash
streamlit run streamlit_app.py
```
## Endpoints

| Endpoint                   | Method | Description                                        |
|----------------------------|--------|----------------------------------------------------|
| `/predict/flood_severity`  | POST   | Predicts flood severity based on user input.      |
| `/predict/economic_impact` | POST   | Estimates economic impact of the flood.           |
| `/predict/water_level`     | POST   | Forecasts water levels based on dam-related parameters. |

## Sample Input

```json
{
    "datetime": "2024-12-16T08:00:00",
    "precipitation_mm": 150.0,
    "dam_discharge_m3_s": 1250.0,
    "state_affected": "Benue",
    "rainfall_mm": 200.0
}
```
## Sample Output
Flood Severity Prediction

```json
{
    "severity_prediction": "Severe",
    "message": "Flood severity is severe. Prepare for immediate action and evacuation if required."
}
```
Economic Impact Prediction
```json
{
    "economic_impact_prediction": 180.0,
    "message": "Economic impact is severe. Widespread damage expected."
}

```
Water Level Prediction
```json
{
    "water_level_prediction": 214.5,
    "message": "Water level is critical. Immediate action required."
}
```






