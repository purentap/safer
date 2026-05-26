import torch
from model import FusionLSTMModel_v2
from omegaconf import DictConfig, OmegaConf
import hydra
import utils 
from datasets import get_dataset_handler
from data import RolloutDataset
from torch.utils.data import DataLoader
from train_utils import eval_epoch

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    model = torch.load("./models/0_model.pth")
    
    model = FusionLSTMModel_v2(cfg.model.params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("./models/0_model.pth")["model_state_dict"])
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

    splitted_rollouts = dataset_handler.split_rollouts(rollouts)
    # Construct datasets and dataloaders from the rollouts
    dataset_by_split_name = {
        k: RolloutDataset(v) 
        for k, v in splitted_rollouts.items()
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

    logs, classification_logs, loss_logs = eval_epoch(model, dataloader_by_split_name,splitted_rollouts, device, batch_size=cfg.training.batch_size, mode="debug", cfg=cfg)

    #print(logs)
    print(f"auc_seen: {logs['auc_by_min_task_step/val_seen']}")
    print(f"auc_unseen: {logs['auc_by_min_task_step/val_unseen']}")
    
    #print(classification_logs)
    
if __name__ == "__main__":
    main()
