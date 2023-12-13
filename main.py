import os
import sys
from tensorflow.keras.models import load_model
from src.build_model import build_model
from src.generate_name import generate_name

if __name__ == "__main__":
   
    # Check if the model file exists
    model_filename = "name_generation_model.h5"

    if os.path.exists(model_filename):
        # Load the existing model
        model = load_model(model_filename)
    else:
        model = build_model()
       
    # Get the seed text from the command-line argument
    if len(sys.argv) > 1:
        seed_text = sys.argv[1]
    else:
        raise "Missing seed argument"
         
    generated_name = generate_name(model, seed_text, length=10)
    print("Generated Name:", generated_name)