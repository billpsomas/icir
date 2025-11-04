"""
Feature extraction and loading utilities for icir dataset.
"""

import os
import csv
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import open_clip


# =============================================================================
# Model Loading
# =============================================================================

def load_model(backbone, device):
    """
    Load vision-language model and tokenizer.
    
    Args:
        backbone: Model name ("clip" or "siglip")
        device: Torch device
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading {backbone} model...")
    
    if backbone == "siglip":
        model, preprocess = open_clip.create_model_from_pretrained("hf-hub:timm/ViT-L-16-SigLIP-256")
        tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-L-16-SigLIP-256")
    elif backbone == "clip":
        model, preprocess = open_clip.create_model_from_pretrained("ViT-L/14", "openai")
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    model.to(device)
    model.eval()

    return {
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer
    }


# =============================================================================
# Text Feature Extraction (for contextualization during feature creation)
# =============================================================================

def text_forward(model, tokenizer, device, batch):
    """Extract and normalize text features for a batch of strings."""
    text_tokens = tokenizer(batch, context_length=model.context_length).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return text_features.detach().to(torch.float32).cpu()


def text_list_to_features(model, tokenizer, text_list, device, batch_size=128):
    """
    Convert list of text strings to normalized features.
    
    Args:
        model: Vision-language model
        tokenizer: Text tokenizer
        text_list: List of text strings
        device: Torch device
        batch_size: Batch size for processing
    
    Returns:
        Tensor of text features (num_texts, dim)
    """
    text_features = []
    num_batches = (len(text_list) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            print(f"Text to features, batch: {i + 1}/{num_batches}", end='\r')
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(text_list))
            batch = text_list[start_idx:end_idx]
            
            if len(batch) > 0:
                batch_features = text_forward(model, tokenizer, device, batch)
                text_features.append(batch_features)
    
    print()  # New line after progress
    return torch.cat(text_features, dim=0).to(device)


def contextualize(model, tokenizer, dim, real_text, corpus_path, number, device, batch_size=512):
    """
    Generate contextualized text features by combining query text with corpus words.
    
    Used during feature extraction to create 'context_text_feats' in addition to 
    standard text features. Combines each query with corpus words in both orders
    and averages the results.
    
    Args:
        model: Vision-language model
        tokenizer: Text tokenizer
        dim: Feature dimension
        real_text: List of query text strings
        corpus_path: Path to CSV file with corpus words
        number: Maximum number of corpus words to use
        device: Torch device
        batch_size: Batch size for processing
    
    Returns:
        Tensor of contextualized text features (num_queries, dim)
    """
    # Load corpus from CSV
    corpus = []
    with open(corpus_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            corpus.append(row[0])
    
    num_queries = len(real_text)
    number = min(number, len(corpus))
    print(f"Contextualizing {num_queries} texts using {number} corpus words")
    
    # Generate "corpus + text" combinations
    combined_texts_1 = [f"{corpus[i]} {real_text[j]}" 
                        for j in range(num_queries) 
                        for i in range(number)]
    features_1 = text_list_to_features(model, tokenizer, combined_texts_1, device, batch_size)
    features_1 = features_1.cpu().view(num_queries, number, dim)
    
    # Generate "text + corpus" combinations
    combined_texts_2 = [f"{real_text[j]} {corpus[i]}" 
                        for j in range(num_queries) 
                        for i in range(number)]
    features_2 = text_list_to_features(model, tokenizer, combined_texts_2, device, batch_size)
    features_2 = features_2.cpu().view(num_queries, number, dim)
    
    # Concatenate and average
    all_features = torch.cat([features_1, features_2], dim=1)
    return all_features.mean(dim=1)

# =============================================================================
# icir Feature Extraction and Loading
# =============================================================================

class icir_dataset(Dataset):
    """
    PyTorch Dataset for ICIR data.
    
    Loads image paths, text queries, and instance IDs from a CSV file.
    Format: image_path,text_query,instance_id
    """
    
    def __init__(self, input_filename, preprocess, root=None):
        """
        Args:
            input_filename: Path to CSV file with format "image_path,text,instance"
            preprocess: Image preprocessing function from model
            root: Optional root directory to prepend to image paths
        """
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        
        # Parse CSV: image_path,text,instance_id
        filenames = [line.strip() for line in lines]
        self.images = [name.split(",")[0] for name in filenames]
        self.text = [name.split(",")[1] for name in filenames]
        self.instance = [name.split(",")[2] for name in filenames]
        self.preprocess = preprocess
        self.root = root
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Returns (image_tensor, img_path, instance_id, text_query)"""
        if self.root is not None:
            img_path = os.path.join(self.root, self.images[idx])
        else:
            img_path = self.images[idx]
        
        image = self.preprocess(Image.open(img_path))
        instance = self.instance[idx]
        text = self.text[idx]
        
        return image, img_path, instance, text

def save_icir(model, tokenizer, dataloader, save_file, device, contextual=None):
    """
    Extract and save icir dataset features.
    
    Args:
        model: Vision-language model
        tokenizer: Text tokenizer
        dataloader: DataLoader for images
        save_file: Path to save pickle file
        device: Torch device
        contextual: Optional corpus for contextualized text features
    """
    all_image_features = []
    all_image_filenames = []
    all_instances = []
    all_texts = []
    all_text_features = []
    
    is_query = "query" in save_file
    total_items = len(dataloader.dataset)
    
    with torch.no_grad():
        for images, img_paths, instances, texts in dataloader:
            current = len(all_image_filenames) + len(img_paths)
            print(f"Processing {current}/{total_items} items...", end="\r")
            
            # Encode images
            images = images.to(device)
            image_features = model.encode_image(images)
            all_image_features.append(image_features.cpu())
            
            # Store metadata
            all_image_filenames.extend(img_paths)
            all_instances.extend(instances)
            all_texts.extend(texts)
            
            # Encode text for query files
            if is_query:
                text_tokens = tokenizer(texts, context_length=model.context_length).to(device)
                text_features = model.encode_text(text_tokens)
                all_text_features.append(text_features.cpu())
    
    print()  # New line after progress
    
    # Build output dictionary
    output = {
        "image_feats": torch.cat(all_image_features, dim=0).numpy(),
        "paths": all_image_filenames,
        "instances": all_instances,
        "texts": all_texts
    }
    
    if is_query:
        output["text_feats"] = torch.cat(all_text_features, dim=0).numpy()
        
        # Add contextualized features if requested
        # use 100 corpus words by default
        if contextual is not None:
            dim = output["text_feats"].shape[1]
            output['context_text_feats'] = contextualize(
                model, tokenizer, dim, all_texts, contextual, 100, device
            ).cpu().numpy()
    else:
        output["text_feats"] = None
    
    # Save to disk
    with open(save_file, "wb") as f:
        pickle.dump(output, f)
    
    print(f"Saved features to {save_file}")

def read_icir(query_dir, database_dir, device, contextualize=False, norm=True, subset=None):
    """
    Load icir dataset features from pickle files.
    
    Args:
        query_dir: Path to query features pickle file
        database_dir: Path to database features pickle file
        device: Torch device to load tensors onto
        contextualize: Use contextualized text features if available (default: False)
        norm: L2-normalize features to unit length (default: True)
        subset: Optional database filtering (not typically used)
    
    Returns:
        Dictionary with structure:
        {
            "query": {
                "image_feats": Tensor (num_queries, dim),
                "text_feats": Tensor (num_queries, dim),
                "paths": List[str],
                "instances": List[str],
                "texts": List[str]
            },
            "database": {
                "image_feats": Tensor (num_database, dim),
                "paths": List[str],
                "instances": List[str],
                "texts": List[str]
            }
        }
    """
    # Load pickle files
    with open(query_dir, "rb") as f:
        query = pickle.load(f)
    with open(database_dir, "rb") as f:
        database = pickle.load(f)
    
    # Convert image features to tensors
    query["image_feats"] = torch.from_numpy(query["image_feats"].astype("float32")).to(device)
    database["image_feats"] = torch.from_numpy(database["image_feats"].astype("float32")).to(device)
    
    # Select text features (contextualized or standard)
    if contextualize and 'context_text_feats' in query:
        query["text_feats"] = torch.from_numpy(query['context_text_feats'].astype("float32")).to(device)
        print("✓ Using contextualized text features")
    else:
        if contextualize:
            print("⚠ Contextualize requested but 'context_text_feats' not found - using standard features")
        query["text_feats"] = torch.from_numpy(query["text_feats"].astype("float32")).to(device)
    
    # Apply subset filtering (if specified - rarely used)
    if subset is not None:
        indices = [i for i, path in enumerate(database['paths']) if "/hn/" not in path]
        database["image_feats"] = database["image_feats"][indices]
        database["paths"] = [database["paths"][i] for i in indices]
        database["instances"] = [database["instances"][i] for i in indices]
        database["texts"] = [database["texts"][i] for i in indices]
    
    print(f"Loaded {query['image_feats'].shape[0]} queries, {database['image_feats'].shape[0]} database items")
    
    # L2-normalize all features to unit length
    if norm:
        query["image_feats"] = query["image_feats"] / query["image_feats"].norm(dim=-1, keepdim=True)
        query["text_feats"] = query["text_feats"] / query["text_feats"].norm(dim=-1, keepdim=True)
        database["image_feats"] = database["image_feats"] / database["image_feats"].norm(dim=-1, keepdim=True)
    
    return {
        "query": {
            "image_feats": query["image_feats"],
            "text_feats": query["text_feats"],
            "paths": query["paths"],
            "instances": query["instances"],
            "texts": query["texts"]
        },
        "database": {
            "image_feats": database["image_feats"],
            "paths": database["paths"],
            "instances": database["instances"],
            "texts": database["texts"]
        }
    }


# =============================================================================
# Text Corpus Feature Extraction and Loading
# =============================================================================


def save_corpus_features(model, tokenizer, corpus_path, save_file, device, batch_size=128):
    """
    Extract and save text corpus features from CSV file.
    
    Args:
        model: Vision-language model
        tokenizer: Text tokenizer
        corpus_path: Path to CSV file with text prompts
        save_file: Path to save pickle file
        device: Torch device
        batch_size: Batch size for processing (default: 128)
    """
    # Read prompts from CSV file
    prompts = []
    with open(corpus_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            prompts.append(row[0])
    
    print(f"Extracting features for {len(prompts)} prompts...")
    
    # Extract text features in batches
    all_text_features = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(prompts))
            batch = prompts[start_idx:end_idx]
            
            print(f"Processing {end_idx}/{len(prompts)} prompts...", end="\r")
            
            text_tokens = tokenizer(batch, context_length=model.context_length).to(device)
            text_features = model.encode_text(text_tokens)
            all_text_features.append(text_features.cpu())
    
    print()  # New line after progress
    
    # Concatenate and save
    all_text_features = torch.cat(all_text_features, dim=0)
    
    output = {
        "feats": all_text_features.numpy(),
        "prompts": prompts
    }
    
    with open(save_file, "wb") as f:
        pickle.dump(output, f)
    
    print(f"Saved {len(prompts)} corpus features to {save_file}")


def read_corpus(pickle_path, device, norm=True):
    """
    Load text corpus features from pickle file.
    
    Args:
        pickle_path: Path to corpus pickle file
        device: Torch device
        norm: L2-normalize features to unit length (default: True)
    
    Returns:
        Tuple of (features, prompts):
            - features: Tensor (num_prompts, dim)
            - prompts: Array of text strings
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    # Convert to tensor
    features = torch.from_numpy(data["feats"].astype("float32")).to(device)
    
    # L2-normalize if requested
    if norm:
        features = features / features.norm(dim=-1, keepdim=True)
    
    prompts = np.array(data["prompts"])
    
    return features, prompts
