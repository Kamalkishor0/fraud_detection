import os
import sys
from pathlib import Path

def get_dataset_path():
    if "google.colab" in sys.modules:
        dataset_path = "/content/creditcard.csv"
        file_id = "16JQSbHah45k3tXGdelSB8sqvcwdIKg47"
        if not os.path.isfile(dataset_path):
            import gdown
            gdown.download(id=file_id, output=dataset_path, quiet=False)
        return dataset_path
    # local: build the path from this file's location, not the notebook's
    return Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"