import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=1000):
    # Base features
    height = np.random.normal(170, 10, n_samples)
    weight = np.random.normal(70, 15, n_samples)
    voice_pitch = np.random.normal(150, 30, n_samples)
    
    # Add BMI as a derived feature
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    # Generate gender with slightly unbalanced distribution
    gender = np.random.binomial(1, 0.48, n_samples)
    
    # Age distribution
    age_probabilities = [0.2, 0.6, 0.2]
    age_group = np.random.choice(3, size=n_samples, p=age_probabilities)
    
    # Apply demographic effects
    # Gender effects
    height[gender == 0] -= np.random.normal(12, 2, sum(gender == 0))
    weight[gender == 0] -= np.random.normal(15, 3, sum(gender == 0))
    voice_pitch[gender == 0] += np.random.normal(80, 10, sum(gender == 0))
    
    # Age effects - Children
    child_mask = age_group == 0
    height[child_mask] = np.random.normal(135, 10, sum(child_mask))
    weight[child_mask] = np.random.normal(32, 8, sum(child_mask))
    voice_pitch[child_mask] += np.random.normal(50, 10, sum(child_mask))
    
    # Seniors
    senior_mask = age_group == 2
    height[senior_mask] -= np.random.normal(3, 1, sum(senior_mask))
    weight[senior_mask] += np.random.normal(2, 1, sum(senior_mask))
    voice_pitch[senior_mask] -= np.random.normal(20, 5, sum(senior_mask))
    
    # Add noise
    noise = np.random.normal(0, 2, n_samples)
    height += noise
    weight += noise * 1.5
    voice_pitch += noise * 3
    
    return pd.DataFrame({
        'height': height,
        'weight': weight,
        'voice_pitch': voice_pitch,
        'bmi': bmi,
        'gender': gender,
        'age_group': age_group
    })

def create_feature_matrix(X):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    height_weight_ratio = X[:, 0] / X[:, 1]
    pitch_height_ratio = X[:, 2] / X[:, 0]
    
    return np.column_stack([X_poly, height_weight_ratio.reshape(-1, 1),
                          pitch_height_ratio.reshape(-1, 1)])

class DemographicPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.gender_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                                 min_samples_split=5, random_state=42)
        self.age_model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                              min_samples_split=5, random_state=42)
    
    def train(self, X, y_gender, y_age):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_enhanced = create_feature_matrix(X_scaled)
        
        # Train models
        self.gender_model.fit(X_enhanced, y_gender)
        self.age_model.fit(X_enhanced, y_age)
    
    def predict(self, height, weight, voice_pitch, bmi):
        # Create input array
        X_input = np.array([[height, weight, voice_pitch, bmi]])
        
        # Scale input
        X_scaled = self.scaler.transform(X_input)
        X_enhanced = create_feature_matrix(X_scaled)
        
        # Make predictions
        gender_pred = self.gender_model.predict(X_enhanced)[0]
        age_pred = self.age_model.predict(X_enhanced)[0]
        
        # Get prediction probabilities
        gender_proba = self.gender_model.predict_proba(X_enhanced)[0]
        age_proba = self.age_model.predict_proba(X_enhanced)[0]
        
        return {
            'gender': 'Female' if gender_pred == 0 else 'Male',
            'gender_probability': max(gender_proba),
            'age_group': ['Child', 'Adult', 'Senior'][age_pred],
            'age_probability': max(age_proba)
        }

def get_user_input():
    try:
        print("\nPlease enter your measurements:")
        height = float(input("Height (in cm): "))
        weight = float(input("Weight (in kg): "))
        voice_pitch = float(input("Voice pitch (in Hz): "))
        
        # Calculate BMI
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        
        return height, weight, voice_pitch, bmi
    
    except ValueError as e:
        print("Please enter valid numerical values.")
        return None

def main():
    try:
        # Generate training data
        print("Generating and training on synthetic data...")
        data = generate_synthetic_data(2000)
        
        # Initialize and train predictor
        predictor = DemographicPredictor()
        X = data[['height', 'weight', 'voice_pitch', 'bmi']]
        predictor.train(X, data['gender'], data['age_group'])
        
        # Get user input and make prediction
        while True:
            measurements = get_user_input()
            if measurements is None:
                continue
                
            height, weight, voice_pitch, bmi = measurements
            
            # Make prediction
            prediction = predictor.predict(height, weight, voice_pitch, bmi)
            
            # Display results
            print("\nPrediction Results:")
            print(f"Gender: {prediction['gender']} (Confidence: {prediction['gender_probability']:.2%})")
            print(f"Age Group: {prediction['age_group']} (Confidence: {prediction['age_probability']:.2%})")
            
            # Ask if user wants to try again
            again = input("\nWould you like to try another prediction? (yes/no): ")
            if again.lower() != 'yes':
                break
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
