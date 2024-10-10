import os
import subprocess

input_dir = "./source"
output_dir = "./source"
for filename in os.listdir(input_dir):
    if filename.endswith(".py"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.html")
        
        # Exporter en HTML
        command = f"marimo export html {input_path} -o {output_path}"
        subprocess.run(command, shell=True, check=True)
        print(f"Exported {filename} to {output_path}")

print("All notebooks have been exported to HTML.")