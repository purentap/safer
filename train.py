import utils as utils 
from model import MLPModel, LSTMModel, FusionLSTMModel
from data import RolloutDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from train_utils import train_epoch, eval_epoch
from torch.optim.lr_scheduler import StepLR
import wandb 
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.seed_everything(cfg.seed)

    ## WANDB CONFIGURATION ##
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    
    # Create group name based on hyperparameters (lr, lambda_reg)
    # This groups runs with same hyperparams but different seeds
    lr_str = f"{cfg.training.learning_rate:.0e}".replace("e-0", "e-")
    lambda_str = f"{cfg.training.lambda_reg:.0e}".replace("e-0", "e-")
    scheduler_str = f"_scheduler_{cfg.scheduler.type}_{cfg.scheduler.eta_min:.0e}".replace("e-0", "e-") if cfg.training.use_scheduler else "No_scheduler"
    epochs_str = f"_epochs_{cfg.training.n_epochs}"
    wandb_group = cfg.wandb.group or f"{cfg.model.type}_lr_{lr_str}_lambda_{lambda_str}_{scheduler_str}{epochs_str}"
    
    # Run name includes seed to distinguish runs within the same group
    wandb_name = cfg.wandb.name or f"seed_{cfg.seed}"
    print(wandb_group, wandb_name)
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, group=wandb_group, config=wandb_config, name=wandb_name)
        wandb.define_metric("auc_by_min_task_step/*", step_metric="epoch", summary="max")
        wandb.define_metric("eval_loss/*", step_metric="epoch", summary="min")
        wandb.define_metric("eval_avg_fail_loss/*", step_metric="epoch", summary="min")
        wandb.define_metric("eval_avg_success_loss/*", step_metric="epoch", summary="min")
        wandb.define_metric("train_loss", step_metric="epoch", summary="min")
        wandb.define_metric("train_fail_succ_loss(wo regularization)", step_metric="epoch", summary="min")
        wandb.define_metric("reg_loss", step_metric="epoch", summary="min")
        wandb.define_metric("train_avg_fail_loss", step_metric="epoch", summary="min")
        wandb.define_metric("train_avg_success_loss", step_metric="epoch", summary="min")
        wandb.run.log_code(root=".")
    ## END OF WANDB CONFIGURATION ##

    ## DATA LOADING ##
    
    rollouts = utils.load_data(cfg.dataset.path)
    splitted_rollouts = utils.split_rollouts(rollouts)

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
    
    ## END OF DATA LOADING ##

    ## MODEL INSTANTIATION ##
    model = utils.create_model(cfg)
    model = model.to(device)
    ## END OF MODEL INSTANTIATION ##

    ## OPTIMIZER INSTANTIATION ##
    optimizer = utils.create_optimizer(model, cfg)
    ## END OF OPTIMIZER INSTANTIATION ##

    ## SCHEDULER INSTANTIATION ##
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10)
    scheduler = utils.create_scheduler(optimizer, cfg)
    print(scheduler)
    ## END OF SCHEDULER INSTANTIATION ##

    pbar = trange(cfg.training.n_epochs)
    train_dataloader = dataloader_by_split_name["train"]
    best_auc_so_far = 0
    best_epoch = 0
    best_val_unseen_auc = 0  # Track val_unseen at best val_seen epoch

    for epoch in pbar:
        model.train()

        loss, reg_loss, fail_succ_loss, avg_fail_loss, avg_success_loss = train_epoch(model, optimizer, train_dataloader, device, cfg.training.lambda_reg, model_type=cfg.model.type)
        pbar.set_description(f"Loss: {loss:.4f}")

        if scheduler:
            scheduler.step()
            #scheduler.step(loss)
        #Evaluation
        model.eval()
        logs, classification_logs, loss_logs = eval_epoch(model, dataloader_by_split_name,splitted_rollouts, device)
        auc_seen = logs["auc_by_min_task_step/val_seen"]
        auc_unseen = logs.get("auc_by_min_task_step/val_unseen", 0)  # Get val_unseen if it exists

                
        if auc_seen > best_auc_so_far:
            best_auc_so_far = auc_seen
            best_epoch = epoch
            best_val_unseen_auc = auc_unseen
            if cfg.training.save_best_model:
                torch.save(model.state_dict(), f"best_model_{wandb_name}.pth")
            if cfg.wandb.enabled:
                wandb.log({"classify_functional_cp/": wandb.Table(dataframe=classification_logs)})
        
        if cfg.wandb.enabled:
            wandb.log({"epoch": epoch})
            wandb.log(logs)
            wandb.log({"train_loss": loss})
            wandb.log({"train_fail_succ_loss(wo regularization)": fail_succ_loss})
            wandb.log({"reg_loss": reg_loss})
            wandb.log({"train_avg_fail_loss": avg_fail_loss})
            wandb.log({"train_avg_success_loss": avg_success_loss})
            wandb.log(loss_logs)
            if scheduler:
                wandb.log({"lr": scheduler.get_last_lr()[0]})
            else:
                wandb.log({"lr": optimizer.param_groups[0]['lr']})
    
    
    if cfg.wandb.enabled:
        wandb.summary["auc_by_min_task_step/val_unseen_at_best_val_seen"] = best_val_unseen_auc
        wandb.summary["best_epoch"] = best_epoch
        wandb.finish()  # Properly end the wandb run        
    return

if __name__ == "__main__":
    main()