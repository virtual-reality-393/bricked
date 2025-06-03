from pathlib import Path
import shutil

DATASET_PATH = r"C:\Users\VirtualReality\Desktop\bricked\model\datasets\brick_figures"


results = list(Path(DATASET_PATH).rglob("*.[tTjJ][xXpP][tTgG]"))

print(results)

for result in results:
    shutil.copy(result,"unpacked_dataset/"+ result.name)