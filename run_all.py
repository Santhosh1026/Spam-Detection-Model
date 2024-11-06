import subprocess

# List of script names to execute
scripts = [
    'data_prep.py',
    'model_training.py',
    'model_testing.py',
    'model_test.py',
    # Add all your script names here
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(['python', script])  # Runs each script one after the other
