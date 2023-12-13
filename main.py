import os
import sys
from src.build_model import build_model
from src.generate_name import generate_name

if __name__ == "__main__":
       
    if len(sys.argv) > 1:
        seed_text = sys.argv[1]
    else:
        raise "Missing seed argument"

    if len(seed_text) != 1:
        raise "Seed argument should be one character"

    model = build_model()
         
    generated_name = generate_name(model, seed_text, length=10)
    print("Generated Name:", generated_name)