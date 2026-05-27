"""
Car Price Prediction - Command Line Interface
Loads trained model and predicts price from user input.
"""

import pandas as pd
import joblib
import os

def load_model():
    model_path = os.path.join(os.getcwd(), "best_car_price_model.pkl")
    if not os.path.exists(model_path):
        print("❌ Model not found. Run 'Optimized version for low-End.py' first.")
        return None
    return joblib.load(model_path)

def get_user_input():
    print("\n" + "="*50)
    print("Enter car details (press Enter to use default values)")
    print("="*50)
    
    # Numeric inputs
    vehicle_age = input("Vehicle age (years) [5]: ")
    vehicle_age = int(vehicle_age) if vehicle_age.strip() else 5
    
    km_driven = input("Kilometers driven [50000]: ")
    km_driven = int(km_driven) if km_driven.strip() else 50000
    
    mileage = input("Mileage (kmpl) [18.0]: ")
    mileage = float(mileage) if mileage.strip() else 18.0
    
    engine = input("Engine (CC) [1200]: ")
    engine = int(engine) if engine.strip() else 1200
    
    max_power = input("Max Power (bhp) [80.0]: ")
    max_power = float(max_power) if max_power.strip() else 80.0
    
    seats = input("Number of seats [5]: ")
    seats = int(seats) if seats.strip() else 5
    
    # Categorical inputs (with defaults)
    brand = input("Brand (e.g., Maruti, Hyundai, Honda) [Maruti]: ")
    brand = brand.strip() if brand.strip() else "Maruti"
    
    model = input("Model (e.g., Swift, i10, City) [Swift]: ")
    model = model.strip() if model.strip() else "Swift"
    
    seller_type = input("Seller type (Dealer / Individual) [Individual]: ")
    seller_type = seller_type.strip() if seller_type.strip() else "Individual"
    
    fuel_type = input("Fuel type (Petrol / Diesel / CNG) [Petrol]: ")
    fuel_type = fuel_type.strip() if fuel_type.strip() else "Petrol"
    
    transmission_type = input("Transmission (Manual / Automatic) [Manual]: ")
    transmission_type = transmission_type.strip() if transmission_type.strip() else "Manual"
    
    # Build DataFrame with exact column names expected by model
    input_data = pd.DataFrame([{
        'brand': brand,
        'model': model,
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'seller_type': seller_type,
        'fuel_type': fuel_type,
        'transmission_type': transmission_type,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }])
    
    return input_data

def main():
    print("🚗 CarDekho Used Car Price Predictor (CLI)")
    model = load_model()
    if model is None:
        return
    
    while True:
        input_df = get_user_input()
        try:
            prediction = model.predict(input_df)[0]
            # Price is likely in Rupees. Convert to Lakhs for readability.
            if prediction > 1e5:
                print(f"\n💰 Estimated Selling Price: ₹{prediction:,.0f} ({(prediction/1e5):.2f} Lakhs)")
            else:
                print(f"\n💰 Estimated Selling Price: ₹{prediction:,.2f}")
        except Exception as e:
            print(f"\n❌ Prediction error: {e}")
            print("Make sure input values are correct.")
        
        again = input("\nPredict another car? (y/n): ").lower()
        if again != 'y':
            break
    
    print("Goodbye!")

if __name__ == "__main__":
    main()