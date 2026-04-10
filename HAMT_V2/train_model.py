import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

from HAMT_model_V2 import HAMTFuelModelV2, MultiTaskLossV2
from feature_engineering import VEDFeatureEngineer


class FuelDataset(Dataset):
    def __init__(self, telemetry, vehicle_context, fuel_loss, behavior_class, route_efficiency):
        self.telemetry = torch.FloatTensor(telemetry)
        self.vehicle_context = torch.FloatTensor(vehicle_context)
        self.fuel_loss = torch.FloatTensor(fuel_loss).unsqueeze(1)
        self.behavior_class = torch.LongTensor(behavior_class)
        self.route_efficiency = torch.FloatTensor(route_efficiency).unsqueeze(1)

    def __len__(self):
        return len(self.telemetry)

    def __getitem__(self, idx):
        return {
            'telemetry': self.telemetry[idx],
            'vehicle_context': self.vehicle_context[idx],
            'fuel_loss': self.fuel_loss[idx],
            'behavior_class': self.behavior_class[idx],
            'route_efficiency': self.route_efficiency[idx],
        }


class HAMTTrainerV2:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.criterion = MultiTaskLossV2()

        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_fuel_loss': [], 'val_fuel_loss': [],
            'train_behavior_loss': [], 'val_behavior_loss': [],
            'train_route_loss': [], 'val_route_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_rmse': [], 'val_rmse': [],
            'train_r2': [], 'val_r2': [],
            'train_mape': [], 'val_mape': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'learning_rates': [],
            'epoch_times': [],
        }

    def train_epoch(self, dataloader, phase='train'):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        epoch_losses = {'fuel': 0, 'behavior': 0, 'driver': 0, 'route': 0, 'total': 0}
        all_fuel_preds, all_fuel_targets = [], []
        all_beh_preds, all_beh_targets = [], []
        all_route_preds, all_route_targets = [], []

        with torch.set_grad_enabled(phase == 'train'):
            for batch in dataloader:
                telemetry = batch['telemetry'].to(self.device)
                vehicle_context = batch['vehicle_context'].to(self.device)

                outputs = self.model(telemetry, vehicle_context)

                targets = {
                    'fuel_loss': batch['fuel_loss'].to(self.device),
                    'behavior_class': batch['behavior_class'].to(self.device),
                    'route_efficiency': batch['route_efficiency'].to(self.device),
                }

                loss, loss_dict = self.criterion(
                    outputs, targets,
                    self.model if phase == 'train' else None
                )

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                for key in epoch_losses:
                    epoch_losses[key] += loss_dict.get(key, 0)

                all_fuel_preds.extend(outputs['fuel_loss'].cpu().detach().numpy().flatten())
                all_fuel_targets.extend(batch['fuel_loss'].numpy().flatten())
                all_beh_preds.extend(
                    torch.argmax(outputs['behavior_logits'], dim=1).cpu().numpy()
                )
                all_beh_targets.extend(batch['behavior_class'].numpy())
                all_route_preds.extend(
                    outputs['route_efficiency'].cpu().detach().numpy().flatten()
                )
                all_route_targets.extend(batch['route_efficiency'].numpy().flatten())

        num_batches = max(len(dataloader), 1)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        fuel_preds = np.array(all_fuel_preds)
        fuel_targets = np.array(all_fuel_targets)
        beh_preds = np.array(all_beh_preds, dtype=np.int64)
        beh_targets = np.array(all_beh_targets, dtype=np.int64)
        route_preds = np.array(all_route_preds)
        route_targets = np.array(all_route_targets)

        mae = mean_absolute_error(fuel_targets, fuel_preds)
        rmse = float(np.sqrt(mean_squared_error(fuel_targets, fuel_preds)))

        if len(set(fuel_targets)) > 1:
            r2 = r2_score(fuel_targets, fuel_preds)
        else:
            r2 = 0.0

        nonzero_mask = np.abs(fuel_targets) > 1e-8
        if nonzero_mask.sum() > 0:
            mape = float(np.mean(np.abs(
                (fuel_targets[nonzero_mask] - fuel_preds[nonzero_mask]) /
                fuel_targets[nonzero_mask]
            )) * 100)
        else:
            mape = 0.0

        accuracy = accuracy_score(beh_targets, beh_preds)
        f1 = f1_score(beh_targets, beh_preds, average='weighted', zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(
            beh_targets, beh_preds, average='weighted', zero_division=0
        )

        return {
            'losses': epoch_losses,
            'mae': float(mae), 'rmse': rmse, 'r2': float(r2), 'mape': mape,
            'accuracy': float(accuracy), 'f1': float(f1),
            'precision': float(precision), 'recall': float(recall),
            'fuel_preds': fuel_preds, 'fuel_targets': fuel_targets,
            'beh_preds': beh_preds, 'beh_targets': beh_targets,
            'route_preds': route_preds, 'route_targets': route_targets,
        }

    def train(self, train_loader, val_loader, epochs=50, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        best_val_mae = float('inf')

        print("\n" + "=" * 80)
        print("TRAINING STARTED")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print("=" * 80)

        for epoch in range(epochs):
            start_time = time.time()

            train_metrics = self.train_epoch(train_loader, 'train')
            val_metrics = self.train_epoch(val_loader, 'val')

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time

            self.history['train_loss'].append(train_metrics['losses']['total'])
            self.history['val_loss'].append(val_metrics['losses']['total'])
            self.history['train_fuel_loss'].append(train_metrics['losses']['fuel'])
            self.history['val_fuel_loss'].append(val_metrics['losses']['fuel'])
            self.history['train_behavior_loss'].append(train_metrics['losses']['behavior'])
            self.history['val_behavior_loss'].append(val_metrics['losses']['behavior'])
            self.history['train_route_loss'].append(train_metrics['losses']['route'])
            self.history['val_route_loss'].append(val_metrics['losses']['route'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['train_r2'].append(train_metrics['r2'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['train_mape'].append(train_metrics['mape'])
            self.history['val_mape'].append(val_metrics['mape'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['learning_rates'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) | LR: {current_lr:.6f}")
                print(f"  Train → Loss: {train_metrics['losses']['total']:.4f} | "
                      f"MAE: {train_metrics['mae']:.4f} | "
                      f"RMSE: {train_metrics['rmse']:.4f} | "
                      f"R²: {train_metrics['r2']:.4f} | "
                      f"Acc: {train_metrics['accuracy']:.4f} | "
                      f"F1: {train_metrics['f1']:.4f}")
                print(f"  Val   → Loss: {val_metrics['losses']['total']:.4f} | "
                      f"MAE: {val_metrics['mae']:.4f} | "
                      f"RMSE: {val_metrics['rmse']:.4f} | "
                      f"R²: {val_metrics['r2']:.4f} | "
                      f"Acc: {val_metrics['accuracy']:.4f} | "
                      f"F1: {val_metrics['f1']:.4f}")

            if val_metrics['losses']['total'] < best_val_loss:
                best_val_loss = val_metrics['losses']['total']
                best_val_mae = val_metrics['mae']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'val_mae': best_val_mae,
                    'val_metrics': {
                        k: val_metrics[k] for k in
                        ['mae', 'rmse', 'r2', 'mape', 'accuracy', 'f1', 'precision', 'recall']
                    }
                }, os.path.join(save_dir, 'hamt_fuel_v2_best.pt'))
                print(f"  Best model saved (loss: {best_val_loss:.4f}, MAE: {best_val_mae:.4f})")

        best_path = os.path.join(save_dir, 'hamt_fuel_v2_best.pt')
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])

        final_val = self.train_epoch(val_loader, 'val')

        self.history['final_val_metrics'] = {
            k: final_val[k] for k in
            ['mae', 'rmse', 'r2', 'mape', 'accuracy', 'f1', 'precision', 'recall']
        }

        self.history['final_predictions'] = {
            'fuel_preds': final_val['fuel_preds'].tolist(),
            'fuel_targets': final_val['fuel_targets'].tolist(),
            'beh_preds': final_val['beh_preds'].tolist(),
            'beh_targets': final_val['beh_targets'].tolist(),
            'route_preds': final_val['route_preds'].tolist(),
            'route_targets': final_val['route_targets'].tolist(),
        }

        self.history['final_confusion_matrix'] = confusion_matrix(
            final_val['beh_targets'], final_val['beh_preds'],
            labels=list(range(6))
        ).tolist()

        try:
            self.history['final_classification_report'] = classification_report(
                final_val['beh_targets'], final_val['beh_preds'],
                labels=list(range(6)),
                target_names=['Eco-Friendly', 'Moderate', 'Aggressive Accel',
                              'Harsh Braking', 'High RPM', 'Stop-and-Go'],
                output_dict=True, zero_division=0
            )
        except Exception:
            self.history['final_classification_report'] = {}

        with open(os.path.join(save_dir, 'training_history_v2.json'), 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print(f"  Best Val Loss:     {best_val_loss:.4f}")
        print(f"  Best Val MAE:      {best_val_mae:.4f}")
        print(f"  Best Val Accuracy: {max(self.history['val_acc']):.4f}")
        print(f"  Best Val F1:       {max(self.history['val_f1']):.4f}")
        print(f"  Best Val R²:       {max(self.history['val_r2']):.4f}")
        print("=" * 80)

        return self.history


def safe_stratified_split(indices, labels, test_size, random_state=42):
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()

    min_required = max(int(np.ceil(1.0 / test_size)), 2)

    if min_count >= min_required:
        return train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=labels
        )
    else:
        print(f" Cannot stratify (min class has {min_count} samples, "
              f"need ≥{min_required}). Using random split.")
        return train_test_split(
            indices, test_size=test_size, random_state=random_state
        )


def train_on_ved_dataset(signals_path, static_path, vehicle_ids,
                         batch_size=32, epochs=50, save_dir='checkpoints'):
    print("=" * 80)
    print("HAMT-Fuel V2 Training Pipeline")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


    print("\n[1/5] Feature Engineering...")
    engineer = VEDFeatureEngineer(window_size=60, overlap=30)
    dataset = engineer.prepare_dataset(signals_path, static_path, vehicle_ids)

    print(f"\n  Dataset Summary:")
    print(f"    Total samples:    {dataset['metadata']['n_samples']}")
    print(f"    Vehicles:         {dataset['metadata']['n_vehicles']}")
    print(f"    Trips:            {dataset['metadata']['n_trips']}")
    print(f"    Telemetry shape:  {dataset['telemetry'].shape}")
    print(f"    Context shape:    {dataset['vehicle_context'].shape}")
    print(f"    Fuel loss range:  [{dataset['fuel_loss'].min():.1f}, {dataset['fuel_loss'].max():.1f}]")
    print(f"    Route eff range:  [{dataset['route_efficiency'].min():.2f}, {dataset['route_efficiency'].max():.2f}]")

    behavior_dist = np.bincount(dataset['behavior_class'], minlength=6)

    print("\n[2/5] Splitting dataset...")

    indices = np.arange(len(dataset['telemetry']))
    labels = dataset['behavior_class']

    trainval_idx, test_idx = safe_stratified_split(
        indices, labels, test_size=0.2, random_state=42
    )

    trainval_labels = labels[trainval_idx]
    train_idx, val_idx = safe_stratified_split(
        trainval_idx, trainval_labels, test_size=0.15, random_state=42
    )

    print(f"    Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"    Train classes: {np.bincount(labels[train_idx], minlength=6).tolist()}")
    print(f"    Val classes:   {np.bincount(labels[val_idx], minlength=6).tolist()}")
    print(f"    Test classes:  {np.bincount(labels[test_idx], minlength=6).tolist()}")

    train_dataset = FuelDataset(
        dataset['telemetry'][train_idx],
        dataset['vehicle_context'][train_idx],
        dataset['fuel_loss'][train_idx],
        dataset['behavior_class'][train_idx],
        dataset['route_efficiency'][train_idx],
    )
    val_dataset = FuelDataset(
        dataset['telemetry'][val_idx],
        dataset['vehicle_context'][val_idx],
        dataset['fuel_loss'][val_idx],
        dataset['behavior_class'][val_idx],
        dataset['route_efficiency'][val_idx],
    )
    test_dataset = FuelDataset(
        dataset['telemetry'][test_idx],
        dataset['vehicle_context'][test_idx],
        dataset['fuel_loss'][test_idx],
        dataset['behavior_class'][test_idx],
        dataset['route_efficiency'][test_idx],
    )

    effective_bs = min(batch_size, len(train_dataset))
    if effective_bs < batch_size:
        print(f" Batch size reduced: {batch_size} → {effective_bs}")

    train_loader = DataLoader(
        train_dataset, batch_size=effective_bs, shuffle=True,
        drop_last=(len(train_dataset) > effective_bs)
    )
    val_loader = DataLoader(val_dataset, batch_size=effective_bs)
    test_loader = DataLoader(test_dataset, batch_size=effective_bs)

    print("\n[3/5] Initializing HAMT-Fuel V2 model...")
    model = HAMTFuelModelV2(
        input_channels=6,
        vehicle_features=7,
        num_behavior_classes=6,
    )
    params = model.count_parameters()
    print(f"    Total parameters: {params['total']:,}")
    for name, count in params['per_module'].items():
        print(f"      {name}: {count:,}")

    print("\n[4/5] Training...")
    trainer = HAMTTrainerV2(model)
    history = trainer.train(train_loader, val_loader, epochs=epochs, save_dir=save_dir)

    print("\n[5/5] Final Test Evaluation...")
    best_path = os.path.join(save_dir, 'hamt_fuel_v2_best.pt')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=trainer.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = trainer.train_epoch(test_loader, 'val')


    print(f"           TEST SET RESULTS              ")
    print(f"MAE:      {test_metrics['mae']:8.4f}% ")
    print(f"RMSE:     {test_metrics['rmse']:8.4f}% ")
    print(f"R²:       {test_metrics['r2']:8.4f}")
    print(f"MAPE:     {test_metrics['mape']:8.2f}%")
    print(f"Accuracy: {test_metrics['accuracy']:8.4f}")
    print(f"F1:       {test_metrics['f1']:8.4f}")
    print(f"Precision:{test_metrics['precision']:8.4f}")
    print(f"Recall:   {test_metrics['recall']:8.4f}")

    history['test_metrics'] = {
        k: float(test_metrics[k]) for k in
        ['mae', 'rmse', 'r2', 'mape', 'accuracy', 'f1', 'precision', 'recall']
    }
    history['test_predictions'] = {
        'fuel_preds': test_metrics['fuel_preds'].tolist(),
        'fuel_targets': test_metrics['fuel_targets'].tolist(),
        'beh_preds': test_metrics['beh_preds'].tolist(),
        'beh_targets': test_metrics['beh_targets'].tolist(),
        'route_preds': test_metrics['route_preds'].tolist(),
        'route_targets': test_metrics['route_targets'].tolist(),
    }
    history['test_confusion_matrix'] = confusion_matrix(
        test_metrics['beh_targets'], test_metrics['beh_preds'],
        labels=list(range(6))
    ).tolist()

    history['dataset_info'] = {
        'n_samples': int(dataset['metadata']['n_samples']),
        'n_vehicles': int(dataset['metadata']['n_vehicles']),
        'n_trips': int(dataset['metadata']['n_trips']),
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
        'behavior_distribution': behavior_dist.tolist(),
        'fuel_loss_stats': {
            'mean': float(dataset['fuel_loss'].mean()),
            'std': float(dataset['fuel_loss'].std()),
            'min': float(dataset['fuel_loss'].min()),
            'max': float(dataset['fuel_loss'].max()),
        },
        'route_efficiency_stats': {
            'mean': float(dataset['route_efficiency'].mean()),
            'std': float(dataset['route_efficiency'].std()),
        }
    }

    history['model_info'] = {
        'total_params': params['total'],
        'trainable_params': params['trainable'],
        'per_module': {k: int(v) for k, v in params['per_module'].items()},
        'architecture': 'HAMT-Fuel V2 (SE + MultiScale-CNN + BiLSTM + GAT + CrossAttn)',
    }

    history_path = os.path.join(save_dir, 'training_history_v2.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\n  Results saved to: {history_path}")
    print("=" * 80)

    return model, history


if __name__ == "__main__":
    signals_path = r"C:\Users\sudup\Desktop\Fuel ML\HAMT_V2\VED_171101_week.csv"
    static_path = r"C:\Users\sudup\Desktop\Fuel ML\HAMT_V2\VED_Static_Data_ICE.xlsx"
    vehicle_ids = [8, 125, 130, 133, 147, 154, 155, 156]

    model, history = train_on_ved_dataset(
        signals_path, static_path, vehicle_ids,
        batch_size=32, epochs=50, save_dir='checkpoints'
    )