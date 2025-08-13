# GrooVAE Model Registry

## 🏆 Production Models

| Alias           | Actual File                | Epoch | Performance | Use Case         | Date Created |
| --------------- | -------------------------- | ----- | ----------- | ---------------- | ------------ |
| `current_best`  | `best_model.pth`           | Best  | Optimal     | Production       | 2024-01-XX   |
| `stable_mid`    | `checkpoint_epoch_50.pth`  | 50    | Good        | Experimentation  | 2024-01-XX   |
| `full_training` | `checkpoint_epoch_100.pth` | 100   | Very Good   | Fine-tuning base | 2024-01-XX   |
| `ultimate`      | `checkpoint_epoch_150.pth` | 150   | Excellent   | Latest training  | 2024-01-XX   |

## 📊 Training Progression

| Epoch | File                       | Size | Loss      | Notes                     |
| ----- | -------------------------- | ---- | --------- | ------------------------- |
| 10    | `checkpoint_epoch_10.pth`  | 52MB | High      | Early foundation          |
| 20    | `checkpoint_epoch_20.pth`  | 52MB | -         | Basic rhythms             |
| 30    | `checkpoint_epoch_30.pth`  | 52MB | -         | Pattern emergence         |
| 40    | `checkpoint_epoch_40.pth`  | 52MB | -         | Groove development        |
| 50    | `checkpoint_epoch_50.pth`  | 52MB | Medium    | Mid-training stable ⭐    |
| 60    | `checkpoint_epoch_60.pth`  | 52MB | -         | Complexity growth         |
| 70    | `checkpoint_epoch_70.pth`  | 52MB | -         | Refinement phase          |
| 80    | `checkpoint_epoch_80.pth`  | 52MB | -         | Advanced patterns         |
| 90    | `checkpoint_epoch_90.pth`  | 52MB | -         | Near convergence          |
| 100   | `checkpoint_epoch_100.pth` | 52MB | Good      | Full training complete ⭐ |
| 110   | `checkpoint_epoch_110.pth` | 52MB | -         | Extended polish           |
| 120   | `checkpoint_epoch_120.pth` | 52MB | -         | Fine tuning               |
| 130   | `checkpoint_epoch_130.pth` | 52MB | -         | Optimization              |
| 140   | `checkpoint_epoch_140.pth` | 52MB | -         | Final refinement          |
| 150   | `checkpoint_epoch_150.pth` | 52MB | Excellent | Ultimate training ⭐      |

## 🔗 Active Aliases

- `current_best.pth` → `best_model.pth`
- `stable_mid.pth` → `checkpoint_epoch_50.pth`
- `full_training.pth` → `checkpoint_epoch_100.pth`
- `ultimate.pth` → `checkpoint_epoch_150.pth`
- `foundation.pth` → `checkpoint_epoch_10.pth`

## 📝 Usage Examples

```bash
# Production inference
python inference.py --model_path saved_models/current_best.pth

# Experimentation
python generate_midi_batch.py --checkpoint saved_models/stable_mid.pth

# Fine-tuning from solid base
python train_finetune.py --base_model saved_models/full_training.pth
```
