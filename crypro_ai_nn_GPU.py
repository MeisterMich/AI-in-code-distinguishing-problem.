import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import gc
import os
import argparse

# Проверка доступности GPU
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Функция для построения индекса файла (без изменений) ----------
def build_index(filename):
    """
    Проходит по файлу, собирает метки и смещения начала каждой строки.
    Возвращает список меток и список смещений.
    """
    labels = []
    offsets = []
    with open(filename, 'r') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label = int(parts[0])
            labels.append(label)
            offsets.append(offset)
    return labels, offsets

# ---------- Датасет с ленивым чтением сырых битов ----------
class RawCodeDataset(Dataset):
    def __init__(self, filename, indices, offsets, n_features):
        self.filename = filename
        self.indices = indices            # список глобальных индексов для этого датасета
        self.offsets = offsets            # полный массив смещений (по глобальному индексу)
        self.n_features = n_features      # количество бит в строке матрицы (n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        offset = self.offsets[global_idx]

        with open(self.filename, 'r') as f:
            f.seek(offset)
            line = f.readline().strip()
            parts = line.split()
            label = int(parts[0])
            bin_strings = parts[1:]

            # Преобразуем строки в тензор битов (float32 для удобства)
            sample = torch.tensor(
                [[int(ch) for ch in bs] for bs in bin_strings],
                dtype=torch.float32
            )
        return sample, torch.tensor(label, dtype=torch.float32)

# ---------- Коллативная функция для сырых данных ----------
def collate_raw(batch):
    samples, labels = zip(*batch)
    # samples: list of (seq_len_i, n_features)
    X_padded = pad_sequence(samples, batch_first=True)   # (batch, max_len, n)
    mask = (X_padded.sum(dim=-1) == 0)                   # (batch, max_len)
    y = torch.stack(labels).view(-1, 1)
    return X_padded, mask, y

# ---------- Модель (DeepDistinguisher с проекцией на борту) ----------
class DeepDistinguisherWithProj(nn.Module):
    def __init__(self, n_features, d_model=1024, nhead=4, num_layers=4, dim_feedforward=4096):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Проекция: 2*n_features -> d_model
        self.proj = nn.Linear(2 * n_features, d_model, bias=False)
        nn.init.kaiming_normal_(self.proj.weight)

        # Трансформер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            activation="gelu",
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, n_features) – биты 0/1
        # Угловое кодирование (p=2)
        angles = 2 * np.pi * x / 2   # p=2
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        # Чередуем cos и sin: (batch, seq_len, 2*n_features)
        encoded = torch.stack([cos_vals, sin_vals], dim=-1).flatten(-2)

        # Линейная проекция
        proj = self.proj(encoded)   # (batch, seq_len, d_model)

        # Позиционное кодирование (векторизовано, на GPU)
        seq_len = proj.size(1)
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device) *
            -(np.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        x = proj + pe.unsqueeze(0)

        # Трансформер
        if mask is not None:
            output = self.transformer(x, src_key_padding_mask=mask)
        else:
            output = self.transformer(x)

        # Max-pooling с маской
        if mask is not None:
            output_masked = output.masked_fill(mask.unsqueeze(-1), -float('inf'))
            pooled, _ = output_masked.max(dim=1)
        else:
            pooled, _ = output.max(dim=1)

        logit = self.fc(pooled)
        return logit

# ---------- Обучение с возможностью возобновления и AMP ----------
def train_with_dataloader(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, device,
                          save_dir='checkpoints', clip_norm=5.0, use_amp=True,
                          start_epoch=0, initial_global_step=0,
                          initial_train_losses=None, initial_val_accuracies=None,
                          initial_lr_history=None, initial_grad_norm_history=None):
    os.makedirs(save_dir, exist_ok=True)
    model.train()
    torch.cuda.empty_cache()
    gc.collect()

    train_losses = initial_train_losses if initial_train_losses is not None else []
    val_accuracies = initial_val_accuracies if initial_val_accuracies is not None else []
    lr_history = initial_lr_history if initial_lr_history is not None else []
    grad_norm_history = initial_grad_norm_history if initial_grad_norm_history is not None else []
    best_val_acc = max(val_accuracies) if val_accuracies else 0.0

    total_steps = len(train_loader) * epochs
    print(f"Total training steps this run: {total_steps}")

    global_step = initial_global_step
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(start_epoch, start_epoch + epochs):
        stime = time()
        total_loss = 0.0
        epoch_grad_norms = []
        epoch_lrs = []

        for X_batch, mask_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+epochs}"):
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(X_batch, mask_batch)
                    loss = loss_fn(logits, y_batch)
                scaler.scale(loss).backward()
            else:
                logits = model(X_batch, mask_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()

            if clip_norm > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                epoch_grad_norms.append(grad_norm.item())

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            global_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            epoch_lrs.append(current_lr)

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        avg_lr = np.mean(epoch_lrs) if epoch_lrs else 0.0
        grad_norm_history.append(avg_grad_norm)
        lr_history.append(avg_lr)

        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, mask_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                mask_batch = mask_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch, mask_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}: train loss = {avg_loss:.4f}, val accuracy = {accuracy:.4f}, "
              f"time spent = {time()-stime:.2f} s.")
        print(f"  Avg learning rate = {avg_lr:.2e}, Avg grad norm = {avg_grad_norm:.4f}")

        # Сохраняем чекпоинт после каждой эпохи
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'val_accuracy': accuracy,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'lr_history': lr_history,
            'grad_norm_history': grad_norm_history
        }
        last_path = os.path.join(save_dir, 'last_checkpoint.pt')
        torch.save(checkpoint, last_path)

        epoch_path = os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pt')
        torch.save(checkpoint, epoch_path)

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> New best model saved with accuracy {accuracy:.4f}")

        model.train()

    final_path = os.path.join(save_dir, 'final_checkpoint.pt')
    torch.save(checkpoint, final_path)
    print(f"Training finished. Total steps: {global_step}")

    return train_losses, val_accuracies, lr_history, grad_norm_history

# ---------- Основная программа ----------
def main():
    parser = argparse.ArgumentParser(description='Train DeepDistinguisher on code data.')
    parser.add_argument('--data', type=str, default='dataset/part_1.txt',
                        help='Path to data file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save final model state dict for future training (e.g., trained_model.pt)')
    args = parser.parse_args()

    # 1. Строим индекс файла (метки и смещения)
    print("Indexing data file...")
    labels, offsets = build_index(args.data)
    print(f"Total samples: {len(labels)}")

    # 2. Определяем размерность n_features по первому образцу
    with open(args.data, 'r') as f:
        f.seek(offsets[0])
        first_line = f.readline().strip()
        parts = first_line.split()
        bin_strings = parts[1:]
        n_rows = len(bin_strings)               # количество строк в матрице (k)
        n_cols = len(bin_strings[0])             # количество бит в строке (n)
    n_features = n_cols
    print(f"Matrix shape per sample: ({n_rows}, {n_cols})")

    # 3. Разбиваем индексы на train/val/test (стратифицированно)
    all_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=42, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, random_state=42, stratify=[labels[i] for i in train_idx]
    )

    # 4. Создаём датасеты (сырые биты)
    train_dataset = RawCodeDataset(args.data, train_idx, offsets, n_features)
    val_dataset = RawCodeDataset(args.data, val_idx, offsets, n_features)
    test_dataset = RawCodeDataset(args.data, test_idx, offsets, n_features)

    # 5. DataLoader'ы с параллельной загрузкой
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_raw, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_raw, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_raw, num_workers=args.num_workers, pin_memory=True
    )

    # 6. Модель и оптимизатор
    d_emb = 1024
    model = DeepDistinguisherWithProj(
        n_features=n_features,
        d_model=d_emb,
        nhead=4,
        num_layers=4,
        dim_feedforward=4096
    )
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)

    warmup_steps = 1000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 1.0
    )

    # Переменные для возобновления
    start_epoch = 0
    initial_global_step = 0
    initial_train_losses = None
    initial_val_accuracies = None
    initial_lr_history = None
    initial_grad_norm_history = None

    if args.resume is not None:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        initial_global_step = checkpoint['global_step']
        initial_train_losses = checkpoint['train_losses']
        initial_val_accuracies = checkpoint['val_accuracies']
        initial_lr_history = checkpoint.get('lr_history', [])
        initial_grad_norm_history = checkpoint.get('grad_norm_history', [])
        print(f"Resumed from epoch {start_epoch}, global step {initial_global_step}")

    # 7. Обучение
    train_losses, val_accuracies, lr_history, grad_norm_history = train_with_dataloader(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler,
        epochs=args.epochs, device=device, save_dir=args.save_dir, clip_norm=5.0,
        use_amp=args.use_amp,
        start_epoch=start_epoch, initial_global_step=initial_global_step,
        initial_train_losses=initial_train_losses, initial_val_accuracies=initial_val_accuracies,
        initial_lr_history=initial_lr_history, initial_grad_norm_history=initial_grad_norm_history
    )

    # 8. Финальная оценка на тесте
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, mask_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch, mask_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.4f}")

    # 9. Сохранение модели для последующего обучения (если указано)
    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)
        print(f"Final model state dict saved to {args.save_model}. You can load it for further training using --resume (with a full checkpoint) or by loading state dict directly.")

    # 10. Визуализация
    epochs_range = range(1, len(train_losses)+1)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss over epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, val_accuracies, 'r-', label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation accuracy over epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, lr_history, 'g-', label='Learning rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning rate over epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, grad_norm_history, 'm-', label='Gradient norm')
    plt.xlabel('Epoch')
    plt.ylabel('Grad norm')
    plt.title('Gradient norm over epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_history_full.png'))
    plt.show()

    # Распределение предсказаний и ROC
    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(all_probs[all_labels == 1], bins=30, alpha=0.7, label='Структурные')
    plt.hist(all_probs[all_labels == 0], bins=30, alpha=0.7, label='Случайные')
    plt.xlabel("Предсказанная вероятность")
    plt.ylabel("Частота")
    plt.legend()
    plt.title("Распределение предсказаний")

    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC-кривая")

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'roc_distribution.png'))
    plt.show()

if __name__ == "__main__":
    main()