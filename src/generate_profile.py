import os
import sys
import subprocess
from pathlib import Path

def create_output_directories():
    """Create output directories if they don't exist"""
    domains = ["beauty", "movies"]
    variants = ["vanilla", "vanilla_structured", "preference", "iterative"]
    
    for domain in domains:
        for variant in variants:
            Path(f"profiles/{domain}/{variant}").mkdir(parents=True, exist_ok=True)

def run_profile_generation():
    """Run all profile generation variants"""
    # Configuration
    variants = {
        "vanilla_b": ("All_Beauty", "vanilla"),
        "vanilla_m": ("Movies_and_TV", "vanilla"),
        "vanilla_Struc_b": ("All_Beauty", "vanilla_structured"),
        "vanilla_Struc_m": ("Movies_and_TV", "vanilla_structured"),
        "pref_b": ("All_Beauty", "preference"),
        "pref_m": ("Movies_and_TV", "preference"),
        "iterative_b": ("All_Beauty", "iterative"),
        "iterative_m": ("Movies_and_TV", "iterative")
    }

    # Create output directories
    create_output_directories()

    # Run each variant
    for script_name, (dataset, variant_type) in variants.items():
        print(f"\nRunning {script_name} for {dataset} - {variant_type}")
        try:
            script_path = f"profile_generator_variants/{script_name}.py"
            if not os.path.exists(script_path):
                print(f"Warning: Script {script_path} not found")
                continue
                
            subprocess.run(["python", script_path], check=True)
            print(f"Successfully completed {script_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_name}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error running {script_name}: {e}")
            continue

if __name__ == "__main__":
    try:
        run_profile_generation()
    except KeyboardInterrupt:
        print("\nProfile generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)