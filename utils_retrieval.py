import torch
import numpy as np
import os
from utils import *
from utils_features import *

def calculate_rankings(args, image_features, text_features, database_features):
    """
    Calculate retrieval rankings using specified method.
    
    Args:
        args: Configuration with method, backbone, and algorithm parameters
        image_features: Query image features (num_queries, dim)
        text_features: Query text features (num_queries, dim)
        database_features: Database image features (num_database, dim)
    
    Returns:
        rankings: Tensor of ranked database indices (num_queries, num_database)
    """
    device = image_features.device
    
    # Features are pre-normalized, but ensure normalization for safety
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    database_features = database_features / database_features.norm(dim=1, keepdim=True)


    # ===== Simple Baseline Methods =====
    if args.method.lower() == "sum":
        sim_img = image_features @ database_features.t()
        sim_text = text_features @ database_features.t()
        total_sim = sim_img + sim_text
        ranks = torch.argsort(total_sim.cpu(), descending=True)
        
    elif args.method.lower() == "text":
        total_sim = text_features @ database_features.t()
        ranks = torch.argsort(total_sim.cpu(), descending=True)
        
    elif args.method.lower() == "image":
        total_sim = image_features @ database_features.t()
        ranks = torch.argsort(total_sim.cpu(), descending=True)
        
    elif args.method.lower() == "product":
        sim_img = image_features @ database_features.t()
        sim_text = text_features @ database_features.t()
        # Rectify similarities (remove negative values)
        sim_img = torch.clamp(sim_img, min=0)
        sim_text = torch.clamp(sim_text, min=0)
        total_sim = sim_img * sim_text
        ranks = torch.argsort(total_sim.cpu(), descending=True)
    
    # ===== Proposed Method (Basic) =====
    # note that this implementation is not as efficient as possible. See paper for details.
    elif "basic" in args.method.lower():

        # Load text corpora for BASIC method
        corpus_dir = os.path.join("features", f"{args.backbone}_features", "corpus")
        
        # Positive corpus (objects)
        pos_corpus_file = os.path.join(corpus_dir, f"{args.specified_corpus}.pkl")
        text_corpus_pos, _ = read_corpus(pos_corpus_file, device, norm=args.norm)
        text_corpus_pos = text_corpus_pos / text_corpus_pos.norm(dim=1, keepdim=True)
        
        # Negative corpus (styles)
        neg_corpus_file = os.path.join(corpus_dir, f"{args.specified_ncorpus}.pkl")
        text_corpus_neg, _ = read_corpus(neg_corpus_file, device, norm=args.norm)
        text_corpus_neg = text_corpus_neg / text_corpus_neg.norm(dim=1, keepdim=True)



        if args.standardize_features:
            
            mean_img = database_features.mean(0, keepdim=True)
            mean_txt = text_corpus_pos.mean(0, keepdim=True)

            if args.use_laion_mean:
                if args.backbone == "clip":
                    with open('./data/laion_mean/laion_1m_mean_clip.pkl', mode='rb') as f:
                        data = pickle.load(f)
                        mean_img = data['laion_1m_mean'].to(device)
                elif args.backbone == "siglip":
                    with open('./data/laion_mean/laion_1m_mean_siglip.pkl', mode='rb') as f:
                        data = pickle.load(f)
                        mean_img = data['laion_1m_mean'].to(device)

            centered_database_features = database_features - mean_img
            centered_image_features = image_features - mean_img

            centered_corpus_pos_features = text_corpus_pos - mean_txt
            centered_corpus_neg_features = text_corpus_neg - mean_txt

            centered_text_features = text_features - mean_txt 
        else:
            centered_database_features, centered_image_features = database_features, image_features
            centered_corpus_pos_features, centered_corpus_neg_features = text_corpus_pos, text_corpus_neg
            centered_text_features = text_features
            
        if args.project_features:
            aa = args.aa
            A, B = centered_corpus_pos_features, centered_corpus_neg_features

            
            # Compute scatter matrices
            Sa = A.T @ A / (A.size(0) - 1)
            Sb = B.T @ B / (B.size(0) - 1)

            C = (1 - aa) * (Sa) - aa * Sb + 1e-5
            L, Vy_t2 = torch.linalg.eigh(C)
            L = L.flip(dims=[0])
            Vy_t = -Vy_t2.flip(dims=[1]).T


            Nc = int(args.num_principal_components_for_projection)
            
            mask_l = L > 0
            Nc = min(Nc, sum(mask_l).item())
            Vy_t = Vy_t[:Nc]

            projection_matrix = Vy_t.T @ Vy_t
        else:
            projection_matrix = torch.eye(centered_database_features.shape[1], device=device)

        # Project features - only one projection is enough (see paper)
        proj_image_features = centered_image_features @ projection_matrix
        proj_database_features = centered_database_features 


        # query expansion
        if args.do_query_expansion:
            #print("Performing query expansion")
            extra_features = centered_database_features
            init_sim_img = proj_image_features @ extra_features.t()
            

            # top K features - hardcoded to 25
            top_values, top_indices = torch.topk(init_sim_img, 25)
            top_features = (centered_database_features).cpu()[top_indices.cpu()]

            # add original features
            top_features = torch.cat((top_features, centered_image_features.unsqueeze(1).cpu()), dim=1)
            # add 1 similarity for original features
            top_values = torch.cat((top_values, torch.ones((top_values.shape[0], 1)).to(device)), dim=1)

            # top values as exponential over cosine similarity
            top_values = torch.exp(.1 * top_values)
            top_values = top_values / top_values.sum(dim=1).unsqueeze(-1)
            
            # weighted mean
            top_features = top_features * top_values.unsqueeze(-1).cpu()
            top_features = top_features.sum(dim=1).to(device)
            
            proj_image_features = (top_features) @ projection_matrix



        sim_img = (proj_image_features) @ (proj_database_features).t()
        sim_img = sim_img.cpu()

        sim_text = centered_text_features @ (centered_database_features).t()
        sim_text = sim_text.cpu()


        if args.normalize_similarities:
            # redundant code inside on-the-fly retrieval, but kept for simplicity
            # one can pre-compute the statistics and normalization factors 

            file_path = os.path.join(args.path_to_synthetic_data, f"dataset_1_sd_{args.backbone}.pkl.npy")
            generated_dataset = np.load(file_path, allow_pickle=True).item()
            image_features_generated = torch.Tensor(generated_dataset['image_features']).to(device)
            text_features_generated = torch.Tensor(generated_dataset['text_features']).to(device)
            
            image_features_generated = image_features_generated / image_features_generated.norm(dim=1, keepdim=True)
            text_features_generated = text_features_generated / text_features_generated.norm(dim=1, keepdim=True)

            if args.standardize_features:
                image_features_generated -= mean_img
                text_features_generated -= mean_txt

            sim_img_gen = image_features_generated @ image_features_generated.t()
            sim_text_gen = text_features_generated @ image_features_generated.t()
            sim_img_min, sim_text_min = sim_img_gen.cpu().min(), sim_text_gen.cpu().min()

            sim_text = (sim_text - sim_text_min)/sim_img_min.abs()        
            sim_img = (sim_img - sim_img_min)/sim_img_min.abs()

        # Rectify similarities (remove negative values)
        sim_text = torch.clamp(sim_text, min=0)
        sim_img = torch.clamp(sim_img, min=0)

        # apply harris criterion
        sim_all = sim_text * sim_img - args.harris_lambda * (sim_text + sim_img)**2

        ranks = torch.argsort(sim_all, descending=True)
    
    return ranks 

def metrics_calc(
    rankings,
    target_domain,
    current_query_classes,
    database_classes,
    database_domains,
    at,
    mode="composed"  # one of: "composed", "image", "text"
):
    metrics = {}

    class_id_map = {class_name: idx for idx, class_name in enumerate(database_classes)}
    domain_id_map = {domain_name: idx for idx, domain_name in enumerate(database_domains)}

    database_classes_ids = [class_id_map[class_name] for class_name in database_classes]
    database_domains_ids = [domain_id_map[domain_name] for domain_name in database_domains]
    query_classes_ids = [class_id_map[class_name] for class_name in current_query_classes]
    target_domain_id = domain_id_map[target_domain]

    device = rankings.device
    database_classes_tensor = torch.tensor(database_classes_ids).to(device)
    database_domains_tensor = torch.tensor(database_domains_ids).to(device)
    query_classes_tensor = torch.tensor(query_classes_ids).to(device)
    target_domain_tensor = torch.tensor(target_domain_id).to(device)

    # Shape: (num_queries, num_database)
    match_class = (database_classes_tensor[rankings] == query_classes_tensor.unsqueeze(1)).float()
    match_domain = (database_domains_tensor[rankings] == target_domain_tensor).float()

    if mode == "image":
        correct = match_class
    elif mode == "text":
        correct = match_domain
    else:
        correct = match_class * match_domain  # composed case

    metrics["mAP"], AP_list = compute_map(correct.cpu().numpy())

    for k in at:
        correct_k = correct[:, :k]
        num_correct = torch.sum(correct_k, dim=1)
        num_predicted = torch.sum(torch.ones_like(correct_k), dim=1)
        num_total = torch.sum(correct, dim=1)

        recall = torch.mean(num_correct / (num_total + 1e-5))
        precision = torch.mean(num_correct / (torch.minimum(num_total, num_predicted) + 1e-5))

        metrics[f"R@{k}"] = round(recall.item() * 100, 2)
        metrics[f"P@{k}"] = round(precision.item() * 100, 2)

    print(metrics)
    return metrics, AP_list

def map_calc_icir(rankings, db_paths, q_paths, q_instances, q_texts, db_instances, db_texts):
    """
    Calculate mean Average Precision for icir dataset.
    
    A match is correct if both the instance AND text query match between query and database item.
    
    Args:
        rankings: Tensor of shape (num_queries, num_database) with ranked database indices
        db_paths: List of database image paths
        q_paths: List of query image paths
        q_instances: List of query instance identifiers
        q_texts: List of query text descriptions
        db_instances: List of database instance identifiers
        db_texts: List of database text descriptions
    
    Returns:
        Dictionary with:
            - "mAP": Mean average precision (float)
            - "APs": List of per-query average precisions
            - "correct_matrix": Binary tensor indicating correct matches (num_queries, num_database)
    """
    num_queries = rankings.shape[0]
    num_database = rankings.shape[1]
    
    # Create binary correctness matrix
    correct_matrix = torch.zeros_like(rankings, dtype=torch.float32)
    
    # Mark correct matches
    for q_idx in range(num_queries):
        query_instance = q_instances[q_idx]
        query_text = q_texts[q_idx]
        
        # Check each ranked database item
        for rank_pos, db_idx in enumerate(rankings[q_idx].tolist()):
            db_instance = db_instances[db_idx]
            db_text = db_texts[db_idx]
            
            # Match requires both instance AND text to match
            if query_instance == db_instance and query_text == db_text:
                correct_matrix[q_idx, rank_pos] = 1.0
    
    # Compute average precision for each query
    mAP, AP_list = compute_map(correct_matrix.cpu().numpy())
    
    return {
        "mAP": mAP,
        "APs": AP_list,
        "correct_matrix": correct_matrix
    }
