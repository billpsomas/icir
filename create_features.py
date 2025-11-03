import os
import numpy as np
import torch
import csv
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from utils_features import *
from utils import *
from pdb import set_trace as st

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, metadata_path=None, transform=None):
        self.img_path = img_path
        self.metadata_path = metadata_path
        self.transform = transform
        self.image_paths = []
        self.texts = []
        if metadata_path is not None:
            with open(metadata_path, 'r') as f:
                reader = csv.reader(f)
                # skip the header
                next(reader)
                # read the rest of the file
                for row in reader:
                    self.image_paths.append(os.path.join(img_path, row[1]))
                    self.texts.append(row[0])
        else:
            for img_file in os.listdir(img_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    self.image_paths.append(os.path.join(img_path, img_file))
                    self.texts.append(img_file)

        #print(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        if self.transform:
            image = self.transform(image)
        return image, text, 'no-domain', self.image_paths[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Frature extraction parameters")
    parser.add_argument(
        "--dataset",
        choices=["ilcir", "corpus"],
        type=str,
        help="define dataset",
    )
    parser.add_argument(
        "--backbone",
        choices=["clip", "siglip"],
        default="clip",
        type=str,
        help="choose the backbone",
    )
    parser.add_argument("--batch", default=512, type=int, help="choose a batch size")
    parser.add_argument(
        "--gpu", default=0, type=int, metavar="gpu", help="Choose a GPU id"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    args.device = setup_device(gpu_id=args.gpu)
    model_struct = load_model(args.backbone, args.device)
    model, preprocess, tokenizer = model_struct["model"], model_struct["preprocess"], model_struct["tokenizer"]

    save_dir = os.path.join("features", f"{args.backbone}_features", args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    if args.dataset.lower() == "corpus":
        corpora_path = "./corpora"
        # list all csv files in the corpora_path
        corpora_names = [f[:-4] for f in os.listdir(corpora_path) if f.endswith('.csv')]
        print("Corpora names:", corpora_names)
        for corpus_name in corpora_names:
            corpus_path = corpora_path + "/" + corpus_name + ".csv"
            save_file = os.path.join(save_dir, corpus_name + ".pkl")
            save_corpus_features(model=model, tokenizer=tokenizer, corpus_path=corpus_path, save_file=save_file, device=args.device)
    elif args.dataset.lower() == "ilcir":
        query_dataset = ilcir_dataset(input_filename=os.path.join(".", "data", args.dataset, "query_files.csv"), preprocess=preprocess, root="./data")
        database_dataset = ilcir_dataset(input_filename=os.path.join(".", "data", args.dataset, "database_files.csv"), preprocess=preprocess, root="./data")
        query_dataloader = DataLoader(query_dataset, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)
        database_dataloader = DataLoader(database_dataset, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)
        save_ilcir(model=model, dataloader=query_dataloader, tokenizer=tokenizer, save_file=os.path.join(save_dir, f"query_{args.dataset}_features.pkl"), device=args.device, contextual="./data/open_image_v7_class_names.csv")
        save_ilcir(model=model, dataloader=database_dataloader, tokenizer=tokenizer, save_file=os.path.join(save_dir, f"database_{args.dataset}_features.pkl"), device=args.device)

if __name__ == "__main__":
    main()
