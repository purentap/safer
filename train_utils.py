import torch
from torch.utils.data import DataLoader
from metrics_utils import eval_roc_auc, eval_functional_conformal
import torch.nn as nn
def calculate_fail_success_loss(losses, valid_masks, success_labels, weights):
    B, T = losses.shape
    # Seq-level aggregation
    fail_mask = success_labels == 0  # (B,)
    success_mask = success_labels == 1  # (B,)
    seq_loss = (losses * valid_masks).sum(-1) / valid_masks.sum(-1)  # (B,)
    success_loss = (success_mask * seq_loss).sum() # scalar
    fail_loss = (fail_mask * seq_loss).sum() # scalar

    # Weight losses according to the input weight and take mean over batch
    loss = weights[0] * fail_loss + weights[1] * success_loss # scalar
    loss = loss / B # scalar

    # Avoid division by zero when computing averages
    fail_count = fail_mask.sum().float()
    success_count = success_mask.sum().float()
    # Use epsilon to avoid division by zero; if count is 0, numerator is also 0, so result is 0
    avg_fail_loss = fail_loss / (fail_count + 1e-8)  # scalar
    avg_success_loss = success_loss / (success_count + 1e-8)  # scalar

    return loss, avg_fail_loss, avg_success_loss


def train_epoch(model, opt, dataloader, device, lambda_reg, model_type):
    use_threshold = False #for now we don't use threshold
    weights = dataloader.dataset.weights #gets per class weights
    batch_losses, reg_losses , fail_succ_losses, avg_fail_losses, avg_success_losses = [], [], [], [], []
    for batch in dataloader:
        img_embeddings = batch["img_embeddings"].to(device)
        action_embeddings = batch["action_embeddings"].to(device)
        valid_masks = batch["valid_masks"].to(device)
        success_labels = batch["success_labels"].to(device)
        
        outputs = model(img_embeddings,action_embeddings)

        outputs = outputs.squeeze(-1) # (B, T)
        
        if model_type == "mlp": #TODO write the corresponding forward passes in the model class
            #LOSS CALCULATION
            #calculate time weights, none is used here
            B, T = valid_masks.shape
            time_weights = torch.ones(B, T).to(valid_masks) # (B, T)
            time_weights = time_weights * valid_masks  # (B, T)
            time_weights = time_weights.to(outputs)
            lower_thresh = 0
            higher_thresh = 50.0
            seq_loss_success = torch.relu(outputs - lower_thresh)  # (B, T)

            if use_threshold:
                seq_loss_fail = time_weights * torch.relu(higher_thresh - outputs)
            else:
                seq_loss_fail = time_weights * (-outputs)
                #print(seq_loss_fail)
            #print((success_labels == 1).float()[:, None]* seq_loss_success + (success_labels == 0).float()[:, None] * seq_loss_fail)
            
            losses = (success_labels == 1).float()[:, None] * seq_loss_success + \
                (success_labels == 0).float()[:, None] * seq_loss_fail  # (B, T)
            
            
        else: #LSTM model 
            B, T, D = batch["img_embeddings"].shape
            # Compute BCE loss on scores at all timesteps
            criterion = nn.BCELoss(reduction="none")
            # Failure is the positive class
            if outputs.isnan().any():
                import pdb; pdb.set_trace()
            failure_labels = (1 - success_labels.float()).unsqueeze(-1).expand_as(outputs)
 
            losses = criterion(outputs, failure_labels) # (B, T)
            '''
            we dont use time weights for now
            # Apply the time weights only on the failure samples
            losses[success_labels == 0] *= time_weights[success_labels == 0] # (B, T)    
            '''

            
        loss, avg_fail_loss, avg_success_loss = calculate_fail_success_loss(losses, valid_masks, success_labels, weights)

        
        
        #regularization loss 
        reg_loss = 0.0 
        for name, param in model.named_parameters():
            if "bias" not in name:
                reg_loss += torch.sum(param ** 2)
        
        reg_loss = lambda_reg * reg_loss   
        reg_losses.append(reg_loss.item())

        ### END OF REGULARIZATION LOSS     
        

        total_loss = loss + reg_loss
        fail_succ_loss = loss
        # Backward and optimize
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        batch_losses.append(total_loss.item())
        fail_succ_losses.append(fail_succ_loss.item())
        avg_fail_losses.append(avg_fail_loss.item())
        avg_success_losses.append(avg_success_loss.item())
        
    losses = sum(batch_losses) / len(batch_losses)
    reg_loss = sum(reg_losses) / len(reg_losses)
    fail_succ_loss = sum(fail_succ_losses) / len(fail_succ_losses)
    avg_fail_loss = sum(avg_fail_losses) / len(avg_fail_losses)
    avg_success_loss = sum(avg_success_losses) / len(avg_success_losses)
    return losses, reg_loss, fail_succ_loss, avg_fail_loss, avg_success_loss
    

def eval_epoch(model, dataloader_by_split_name, rollouts_by_split_name, device, batch_size=64):
    scores_by_split_name = {}
    loss_logs = {}
    for split, dataloader in dataloader_by_split_name.items():
        #re-create dataloader to disable shuffling
        dataloader = DataLoader(dataloader.dataset, batch_size = batch_size, shuffle=False, num_workers =0)
        
        weights = dataloader.dataset.weights #gets per class weights

        scores = []
        all_valid_masks = []
        all_labels = []
        batch_losses, avg_fail_losses, avg_success_losses = [], [], []
        criterion = nn.BCELoss(reduction="none")
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

                failure_labels = (1 - success_labels.float()).unsqueeze(-1).expand_as(outputs)
                batch_loss = criterion(outputs, failure_labels) # (B, T)
                batch_loss, batch_avg_fail_loss, batch_avg_success_loss = calculate_fail_success_loss(batch_loss, valid_masks, success_labels, weights)
                batch_losses.append(batch_loss.item())
                avg_fail_losses.append(batch_avg_fail_loss.item())
                avg_success_losses.append(batch_avg_success_loss.item())


            loss = sum(batch_losses) / len(batch_losses)
            avg_fail_loss = sum(avg_fail_losses) / len(avg_fail_losses)
            avg_success_loss = sum(avg_success_losses) / len(avg_success_losses)

            scores = torch.cat(scores, dim=0).squeeze(-1)
            all_valid_masks = torch.cat(all_valid_masks, dim=0) #(dataloader_size, T)
            all_labels = torch.cat(all_labels, dim=0) #(dataloader_size,)
        scores = scores.detach().cpu().numpy()
        seq_lengths = all_valid_masks.sum(dim=-1).cpu().numpy() # (B,)
        scores_by_split_name[split] = [scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))]
        if split != "train":
            loss_logs[f"eval_loss/{split}_loss"] = loss
            loss_logs[f"eval_loss/{split}_avg_fail_loss"] = avg_fail_loss
            loss_logs[f"eval_loss/{split}_avg_success_loss"] = avg_success_loss

    logs= eval_roc_auc(scores_by_split_name, rollouts_by_split_name)
    classification_logs = eval_functional_conformal(scores_by_split_name, rollouts_by_split_name, calib_split_names = ["val_seen"], test_split_names = ["val_unseen"])
    return logs, classification_logs, loss_logs

