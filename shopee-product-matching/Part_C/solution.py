"""Part C: ResNet image embedding extractor."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import models

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image-dir",required=True)
    ap.add_argument("--output",default="image_embeddings.npy")
    ap.add_argument("--max-images",type=int,default=5000)
    args=ap.parse_args()

    device="cuda" if torch.cuda.is_available() else "cpu"
    weights=models.ResNet18_Weights.DEFAULT
    model=models.resnet18(weights=weights)
    model.fc=torch.nn.Identity()
    model.eval().to(device)
    transform=weights.transforms()

    paths=[p for p in Path(args.image_dir).iterdir() if p.is_file()][:args.max_images]
    features=[]; kept=[]
    with torch.no_grad():
        for path in paths:
            try:
                image=Image.open(path).convert("RGB")
                x=transform(image).unsqueeze(0).to(device)
                z=torch.nn.functional.normalize(model(x),dim=1)
                features.append(z.cpu().numpy()[0]); kept.append(str(path))
            except Exception as exc:
                print("skip",path,exc)
    E=np.asarray(features,dtype=np.float32)
    np.save(args.output,E)
    Path(args.output).with_suffix(".paths.txt").write_text("\n".join(kept),encoding="utf-8")
    print("embedding shape:",E.shape)

if __name__=="__main__":
    main()
