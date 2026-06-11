import torch
from model import FusionLSTMModel_v2
from omegaconf import DictConfig, OmegaConf
import hydra
import utils 
from datasets import get_dataset_handler
from data import RolloutDataset
from torch.utils.data import DataLoader
from train_utils import eval_epoch
import pandas as pd
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    model_path = "/home/enes/puren/research/fail_detection_embeddings/models/openvla_widowx/1_model.pth"
    model = torch.load(model_path)
    print(model["epoch"])
    model = FusionLSTMModel_v2(cfg.model.params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.load(model_path)["model_state_dict"])
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(model)

    #Prepare dataset
    utils.seed_everything(0)
    dataset_name = cfg.dataset.get("name", None) if hasattr(cfg, "dataset") else None
    print(dataset_name)
    DatasetHandler = get_dataset_handler(dataset_name)
    dataset_handler = DatasetHandler(cfg)

    rollouts = dataset_handler.load_rollouts()

    print(rollouts[0].get_img_embeddings().dtype)
    print(rollouts[0].get_action_embeddings().dtype)
    utils.seed_everything(cfg.seed)

    rollouts_by_split_name = dataset_handler.split_rollouts(rollouts)
    # Construct datasets and dataloaders from the rollouts
    dataset_by_split_name = {
        k: RolloutDataset(v) 
        for k, v in rollouts_by_split_name.items()
    }
    dataloader_by_split_name = {
    k: DataLoader(
        v, 
        batch_size=cfg.training.batch_size, 
        shuffle="train" in k, 
        num_workers=0)
        for k, v in dataset_by_split_name.items()
    }
    # end of data prep

    scores_by_split_name={}
    labels_by_split_name={  }
    for split, dataloader in dataloader_by_split_name.items():
        #re-create dataloader to disable shuffling
        dataloader = DataLoader(dataloader.dataset, batch_size = 64, shuffle=False, num_workers =0)
        

        weights = dataloader.dataset.weights #gets per class weights

        scores = []
        all_valid_masks=[]
        all_labels=[]
        with torch.no_grad():
            for batch in dataloader:
                img_embeddings = batch["img_embeddings"].to(device)
                action_embeddings = batch["action_embeddings"].to(device)
                success_labels = batch["success_labels"].to(device)
                valid_masks = batch["valid_masks"].to(device)
                all_valid_masks.append(valid_masks)
                all_labels.append(success_labels)
                outputs = model(img_embeddings, action_embeddings)
                outputs = outputs.squeeze(-1) # (B, T)

                scores.append(outputs)

        scores = torch.cat(scores, dim=0).squeeze(-1)
        all_valid_masks = torch.cat(all_valid_masks, dim=0) #(dataloader_size, T)

        all_labels = torch.cat(all_labels, dim=0) #(dataloader_size,)
        scores = scores.detach().cpu().numpy()
        seq_lengths = all_valid_masks.sum(dim=-1).cpu().numpy() # (B,)
        scores_by_split_name[split] = [scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))]
        labels_by_split_name[split] = [1 - all_labels[i].cpu().numpy() for i in range(len(all_labels))] #
    
        #TODO ALT KISIM KALKACAK
    
    folder_path = "/home/enes/puren/data/openvla_widowx/correct_embeddings/openvla_widowx_correct_embeds_data/openvla_widowx/"
    df = []
    for split in scores_by_split_name.keys():        
        scores = scores_by_split_name[split]
        labels = labels_by_split_name[split]
        rollouts = rollouts_by_split_name[split]

        for i in range(len(scores)):
            rollout = rollouts[i]
            mp4_path = rollout.mp4_path

            mp4_path = mp4_path.replace("rollouts/", "")
            mp4_path = folder_path + mp4_path
            row = {}
            row["output_scores"] = scores[i]
            row["failure_labels"] = labels[i]
            row["split"] = split
            row["threshold"] = 0.5
            row["mp4_path"] = mp4_path
            df.append(row)            
    df = pd.DataFrame(df)
    df.to_excel("openvla_widowx_outputs_seed1.xlsx", index=False)

    #print(df)
    logs, classification_logs, loss_logs, classification_logs_fixed_threshold, classification_logs_best_threshold= eval_epoch(model, dataloader_by_split_name,rollouts_by_split_name, device, batch_size=cfg.training.batch_size, mode=None, cfg=cfg)

    #print(logs)
    #print(f"auc_seen: {logs['auc_by_min_task_step/val_seen']}")
    #print(f"auc_unseen: {logs['auc_by_min_task_step/val_unseen']}")
    
    print(classification_logs_fixed_threshold)
    
if __name__ == "__main__":
    main()
