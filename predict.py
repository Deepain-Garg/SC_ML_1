import os
import pickle
import argparse
import sys
import pandas as pd

MODEL_PATH = "house_price_model.pkl"

def get_interactive_input():
    print("\n--- Enter House Details for Prediction ---")
    try:
        sqft = float(input("Enter Square Footage (e.g., 2500): "))
        if sqft <= 0:
            raise ValueError("Square footage must be a positive number.")
            
        beds = float(input("Enter Number of Bedrooms (e.g., 4): "))
        if beds < 0:
            raise ValueError("Number of bedrooms cannot be negative.")
            
        baths = float(input("Enter Number of Bathrooms (e.g., 3): "))
        if baths < 0:
            raise ValueError("Number of bathrooms cannot be negative.")
            
        return sqft, beds, baths
    except ValueError as e:
        print(f"Error: Invalid input ({e}). Please try again.")
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Predict House Prices using Trained Linear Regression Model")
    parser.add_argument("--square_feet", type=float, help="Square footage of the house")
    parser.add_argument("--bedrooms", type=float, help="Number of bedrooms")
    parser.add_argument("--bathrooms", type=float, help="Number of bathrooms")
    
    args = parser.parse_args()

    # Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model '{MODEL_PATH}' not found.")
        print("Please run the training script first: python train.py")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Determine input method
    if args.square_feet is not None or args.bedrooms is not None or args.bathrooms is not None:
        # Command line arguments provided, ensure all three are present
        if args.square_feet is None or args.bedrooms is None or args.bathrooms is None:
            print("Error: When using arguments, you must provide all three features:")
            print("  --square_feet, --bedrooms, and --bathrooms")
            sys.exit(1)
        sqft, beds, baths = args.square_feet, args.bedrooms, args.bathrooms
    else:
        # Fall back to interactive mode if no arguments are provided
        sqft, beds, baths = get_interactive_input()

    # Run inference
    # Wrap in a DataFrame with correct feature names to avoid user warnings
    features = pd.DataFrame(
        [[sqft, beds, baths]], 
        columns=['SquareFeet', 'Bedrooms', 'Bathrooms']
    )
    predicted_price = model.predict(features)[0]

    # Display result
    print("\n" + "=" * 50)
    print("                 PREDICTION RESULTS                 ")
    print("=" * 50)
    print(f"  Inputs:")
    print(f"    - Square Footage:   {sqft:,.2f} sq ft")
    print(f"    - Bedrooms:         {beds:.1f}")
    print(f"    - Bathrooms:        {baths:.1f}")
    print("-" * 50)
    print(f"  Predicted House Price: {predicted_price:.4f} Lakhs")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
