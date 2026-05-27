# 🚗 CarDekho Used Car Price Predictor
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange.svg)](https://scikit-learn.org/)
[![R² Score](https://img.shields.io/badge/R²-0.93-brightgreen)](./model_metrics.json)

Machine learning model to predict **used car selling prices** using the CarDekho dataset.  
Random Forest Regressor achieves **R² = 0.93** on test data. Includes data cleaning, feature importance analysis, and a command-line prediction tool.

---

## 📌 Table of Contents
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Requirements](#requirements)
- [How to Run – Step by Step](#how-to-run--step-by-step)
- [1️⃣ Train the Model](#1️⃣-train-the-model)
- [2️⃣ Predict a Car’s Price](#2️⃣-predict-a-cars-price)
- [Project File Structure](#project-file-structure)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📁 Dataset

- **Source**: [GitHub – manishkr1754/CarDekho_Used_Car_Price_Prediction](https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv)
- **Rows**: 15,411
- **Features**:
  - `brand`, `model`, `vehicle_age`, `km_driven`, `seller_type`
  - `fuel_type`, `transmission_type`, `mileage`, `engine`, `max_power`, `seats`
- **Target**: `selling_price` (in Rupees)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9301 |
| **MAE** | ₹1,02,373 (~1.02 Lakhs) |
| **RMSE** | ₹2,29,354 |

**Top 5 important features** (from `feature_importance.png`):
1. `max_power` – 65%
2. `vehicle_age` – 15%
3. `km_driven` – 8%
4. `engine` – 5%
5. `mileage` – 5%

---

## 🛠️ Requirements

Install dependencies once:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```
# 🚗 Car Price Prediction Model – CarDekho Dataset

## 📌 Project Overview
This project builds a machine learning model to predict the **selling price** of used cars using the **CarDekho dataset**.  
It includes:

- Data cleaning & feature engineering  
- Model training (Random Forest) with **R² = 0.93**  
- Command‑line prediction tool (no web interface)  
- Visualisations: actual vs predicted, residual plot, feature importance  

**Perfect for a portfolio / academic submission.**

---

## 📁 Dataset
- **Source**: [GitHub – manishkr1754/CarDekho_Used_Car_Price_Prediction](https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv)  
- **Rows**: 15,411  
- **Features**:
  - `brand`, `model`, `vehicle_age`, `km_driven`, `seller_type`, `fuel_type`, `transmission_type`, `mileage`, `engine`, `max_power`, `seats`
- **Target**: `selling_price` (in Rupees)

---

## 🧠 Model & Performance
- **Algorithm**: Random Forest Regressor  
- **Preprocessing**:  
  - Standard scaling for numeric features  
  - One‑hot encoding for categorical features  
  - Top‑10 categories for `model` to avoid explosion  
- **Performance** (on test set):
  - R² = **0.9301**  
  - MAE = **₹1,02,373** (~1.02 Lakhs)  
  - RMSE = **₹2,29,354**  

---

## 🛠️ Requirements (install once)
Make sure Python 3.8+ is installed, then run:
---
## Windows:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```
---
## Linux
```bash
pip install pandas numpy scikit-learn matplotlib joblib
#For Ubuntu or in case of problems
sudo apt install python3-matplotlib python3-joblib python3-sklearn pyhton3-numpy python3-pandas
```
---
## Mac 
```bash
pip3 install scikit-learn pandas numpy
```
### Alternate Method(Using HomeBrew):
1.Install HomeBrew(If You Haven't Already):
```bash
/bin/bash -c "$(curl -fsSL https://githubusercontent.com)"
```
2.Install The Libraries
```bash
brew install numpy pandas scikit-learn
```
---
# 🚀 How to Run – Step by Step
## 1️⃣ Train the Model (only once)
```bash
python "Optimized version for low-End.py"
```
### What happens?
```text
Downloads the dataset automatically
Cleans & prepares the data
Trains a Random Forest model
Saves:best_car_price_model.pkl – trained model
model_metrics.json – R², MAE, RMSE
car_price_results.png – actual vs predicted + residual plot
feature_importance.png – top 10 feature importances
⏱️ Training takes 2–5 minutes on a modern laptop or Google Colab.
```
## 2️⃣ Predict a Car’s Price
```bash
python predict_cli.py
```
### You will be asked for car details.
#### Example interaction:
````text
Vehicle age (years) [5]: 4
Kilometers driven [50000]: 35000
Mileage (kmpl) [18.0]: 22.5
Engine (CC) [1200]: 1197
Max Power (bhp) [80.0]: 82
Seats [5]: 5
Brand [Maruti]: Maruti
Model [Swift]: Baleno
Seller type (Dealer/Individual) [Individual]: Individual
Fuel type (Petrol/Diesel/CNG) [Petrol]: Petrol
Transmission (Manual/Automatic) [Manual]: Manual

### Output:
💰 Estimated Selling Price: ₹4,52,300 (4.52 Lakhs)
You can predict as many cars as you want in the same session.
````
