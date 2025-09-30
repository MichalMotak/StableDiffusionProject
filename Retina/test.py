# build_dataset.py
from datasets import Dataset
from pathlib import Path
import pandas as pd
import os
from natsort import natsorted
# root = Path("/home/MichalMo/projects/StableDiffusion/Retina/dataset/cadis")
# img_dir = root / "images"
# depth_dir = root / "depth"
# mask_dir = root / "mask"



# files = sorted([p.name for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

# print(files)

# rows = []
# for fn in files:
#     rows.append({
#         "image_path": str(img_dir / fn),
#         "mask_path": str(mask_dir / fn),
#         "depth_path": str(depth_dir / fn),

#         "text": ""   # <-- empty prompt for every example
#     })

# df = pd.DataFrame(rows)
# ds = Dataset.from_pandas(df)
# ds.to_csv(str(root / "pairs.csv"), index=False)   # optional save
# print("Dataset created with", len(ds), "rows")



root = Path("/home/MichalMo/projects/SurGrID/datasets/cadis/CaDISv2")
img_dir = root / "images"
depth_dir = root / "depth"
#mask_dir = root / "mask"

rows = []

for f in root.iterdir():
    print(f)

    img_dir = os.path.join(root,f, "Images")
    depth_dir = os.path.join(root,f, "Labels")
    #mask_dir = os.path.join(root,f, "mask")

    files = natsorted([p for p in os.listdir(img_dir) if p[-3:] == "png" ])

    # print(files)

    for fn in files:
        rows.append({
            "image_path": os.path.join(img_dir , fn),
            "depth_path": os.path.join(depth_dir , fn),

            "text": ""   # <-- empty prompt for every example
        })

df = pd.DataFrame(rows)
ds = Dataset.from_pandas(df)
ds.to_csv("/home/MichalMo/projects/StableDiffusion/Retina/dataset/cadis/pairs.csv", index=False)   # optional save
print("Dataset created with", len(ds), "rows")