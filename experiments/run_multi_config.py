import sys
import os
import experiments.experiment_runner as expR

dir = sys.argv[1]

if not os.path.isdir("experiments/" + dir):
    print("not a folder")
    sys.exit(0)

for config_file in os.listdir("experiments/" + dir):
    if os.path.isdir(config_file):
        continue
    output_dir = expR.run_experiment("experiments/" + f"{dir}/{config_file}" )
    print(f"\nResults saved to: {output_dir}")

