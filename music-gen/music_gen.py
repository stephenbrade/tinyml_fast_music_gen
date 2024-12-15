import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import math

from tqdm import tqdm
from .args import parse_args
from .dataset import create_epiano_dataloaders, compute_epiano_accuracy, VOCAB_SIZE, TOKEN_PAD, jsb_chorales_dataloaders
from .lr_scheduler import LrStepTracker
sys.path.append('/storage/vsub851/neural-architecture-search/music-gen/MusicTransformer-Pytorch')
from model.music_transformer import MusicTransformer
from .lstm import MusicLSTM
from rep_sim import rep_similarity_loss

def total_loss(train_model, target_model, rep_sim, loss_fn, preds, imgs, labels, rep_sim_alpha, device, student_model = 'Music-LSTM'):
    rep_sim = rep_similarity_loss(train_model, target_model, rep_sim, imgs, device, student_model = student_model)
    ce_loss = loss_fn(preds, labels)
    return ce_loss + rep_sim_alpha * rep_sim, rep_sim, ce_loss

def get_ppl(model, test_loader, loss_fn, device):
    total_loss = 0
    for batch in tqdm(test_loader, desc = 'Iterating over test loader'):
        input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
    return math.exp(total_loss / len(test_loader))

def avg_step_size(model, before_state_dict):
    sum_changes = 0
    count = 0
    with torch.no_grad():
        after_state_dict = model.state_dict()
        for key in before_state_dict:
            change = (after_state_dict[key] - before_state_dict[key]).abs().mean().item()
            sum_changes += change
            count += 1
    return sum_changes / count

def eval_music(test_loader, model, device):
    model = model.eval()
    sum_acc = 0
    total_eval = len(test_loader)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc = 'Evaluating accuracy...'):
            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            logits = model(input_ids)
            sum_acc += float(compute_epiano_accuracy(logits, labels))
        avg_acc = sum_acc / total_eval
    return avg_acc

def guidance_music(args, exp_name, task, repr_sim, student_model, target_model = 'music-trans', pretrained = True, num_epochs = 10, 
             batch_size = 64, num_workers = 16, lr = 1e-3, accumulation = 1, embedding_dim = 768, hidden_dim = 768, num_layers = 3, fc_dim = 512,
             rep_dist = None, rep_sim_alpha = 1.0):
    wandb.init(
        project = exp_name,
        config = {
            'model': student_model,
            'repr-sim': repr_sim,
            'lr': lr,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'fc_dim': fc_dim,
            'rep_dist': rep_dist
        }
    )

    if task == 'maestro':
        train_dataloader, valid_dataloader, test_dataloader = create_epiano_dataloaders('music-gen/chunked-maestro-v2.0.0/', 1024, batch_size, num_workers)
        vocab_size = VOCAB_SIZE
    else:
        train_dataloader, valid_dataloader, test_dataloader = jsb_chorales_dataloaders('music-gen/JSB-Chorales-dataset/Jsb16thSeparated.json', 1024, True, batch_size, num_workers)
        vocab_size = 92
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_fn = nn.CrossEntropyLoss(ignore_index = -1)

    if student_model == 'Music-LSTM':
        model = MusicLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, fc_dim, device)
    elif student_model == 'Music-Trans':
        model = MusicTransformer(vocab_size, n_layers=3, num_heads=12,
                    d_model=768, dim_feedforward=2048, dropout=0.1,
                    max_sequence=1024, rpr=True)
    else:
        raise NotImplementedError()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr = lr)
    lr_stepper = LrStepTracker(hidden_dim)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_stepper.step)
    val_losses = []
    epoch_train_losses = []
    step_train_losses = []
    step_sizes = []
    step_ce_loss = []
    step_rep_sim_loss = []
    accs = []

    if repr_sim:
        if target_model == 'music-trans':
            target_model = MusicTransformer(vocab_size, n_layers=3, num_heads=12,
                        d_model=768, dim_feedforward=2048, dropout=0.1,
                        max_sequence=1024, rpr=True)
            if pretrained:
                target_model.load_state_dict(torch.load('saved_models/music_trans.pt'))
        target_model = target_model.to(device)

    for epoch in range(num_epochs):
        model = model.eval()
        valid_loss = 0.0
        # hidden = None
        for i, batch in enumerate(tqdm(valid_dataloader, desc = 'Iterating over valid loader...')):
            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            with torch.no_grad():
                logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = loss_fn(logits, labels)
            valid_loss += loss.item()
        avg_val_loss = valid_loss / len(valid_dataloader)
        wandb.log({'val_loss': avg_val_loss})
        val_losses.append(avg_val_loss)
        if avg_val_loss <= min(val_losses):
            torch.save(model.state_dict(), f'saved_models/{exp_name}.pt')
        print(f'Epoch {epoch + 1}, Validation Loss {avg_val_loss}')
        acc = eval_music(test_dataloader, model, device)
        print(f'Epoch {epoch + 1}, Testing Accuracy {acc}')
        accs.append(acc)
        model = model.train()
        train_loss = 0.0
        # hidden = None
        for i, batch in enumerate(tqdm(train_dataloader, desc = 'Iterating over train loader...')):
            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            input_ids[input_ids == -1] = VOCAB_SIZE
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            if not repr_sim:
                loss = loss_fn(logits, labels)
                ce_loss = None
            else:
                loss, sim_loss, ce_loss = total_loss(model, target_model, rep_dist, loss_fn, logits, input_ids, labels, rep_sim_alpha, 
                                                    device, student_model = student_model)
                step_ce_loss.append(ce_loss.item())
                step_rep_sim_loss.append(sim_loss.item())
                if i % 20 == 0:
                    avg_ce_loss = np.mean(step_ce_loss[-20:])
                    avg_rep_sim_loss = np.mean(step_rep_sim_loss[-20:])
                    wandb.log({'ce_loss': avg_ce_loss, 'rep_sim_loss': avg_rep_sim_loss})
            before_update_params = {name: param.clone() for name, param in model.named_parameters()}
            loss.backward()   
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.25)
            if (i + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            if ce_loss == None:
                train_loss += loss.item()
            else:
                train_loss += ce_loss.item()
            step_train_losses.append(loss.item())
            # hidden = tuple(h.detach() for h in hidden)

            step_size = avg_step_size(model, before_update_params)
            step_sizes.append(step_size)

            if i % 5 == 0:
                avg_train_loss = np.mean(step_train_losses[-20:])
                wandb.log({'train_loss': avg_train_loss, 'step_size': step_size})
        avg_train_loss = train_loss / len(train_dataloader)
        epoch_train_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1}, Training Loss {avg_train_loss}')

    if not os.path.exists(f'{args.logging}/{args.exp_name}'):
        os.makedirs(f'{args.logging}/{args.exp_name}')
    with open(f'{args.logging}/{exp_name}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    loss_info = {'step_train_losses': step_train_losses, 'step_sizes': step_sizes, 'val_losses': val_losses, 'epoch_train_losses': epoch_train_losses, 'step_ce_loss': step_ce_loss, 'step_rep_sim_loss': step_rep_sim_loss, 'accuracies': accs}
    loss_info = {key: value for key, value in loss_info.items() if value != []}
    with open(f'{args.logging}/{exp_name}/info.json', 'w') as f:
        json.dump(loss_info, f)

    wandb.finish()
    return model, epoch_train_losses, step_train_losses, val_losses, step_ce_loss, step_rep_sim_loss

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.eval:
        model, epoch_train_losses, step_train_losses, val_losses, step_ce_loss, step_rep_sim_loss = guidance_music(args, args.exp_name, args.task, args.rep_sim, 
                                                                                                                args.student_model, args.target_model, args.pretrained, args.num_epochs, 
                                                                                                                args.batch_size, args.num_workers, args.lr, args.accumulation, args.embedding_dim, args.hidden_dim, args.num_layers, 
                                                                                                                args.fc_dim, args.rep_dist, args.rep_sim_alpha)