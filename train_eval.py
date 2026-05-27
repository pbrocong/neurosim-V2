# train_eval.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _record_reads_if_possible(optimizer, n_samples: int):
    """Optimizer-level read tracking for energy accounting.

    Each forward pass reads every weight once per sample. We treat
    `n_reads = n_samples * total_weights`.
    """
    if hasattr(optimizer, "record_reads") and hasattr(optimizer, "total_weights"):
        optimizer.record_reads(int(n_samples) * int(optimizer.total_weights))


def _set_bn_eval(model: nn.Module):
    """Force every BatchNorm{1,2,3}d into eval mode (uses running stats).

    Needed by `train_online()` because BN with batch=1 in train mode is
    degenerate (variance=0 → div-by-zero / NaN gradients).
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            n += 1
    return n


# ---------------------------------------------------------------------------
# Mini-batch training (default)
# ---------------------------------------------------------------------------
def train(model, device, train_loader, optimizer, epoch, loss_mode="ce"):
    """Standard mini-batch training. Returns train accuracy (%)."""
    model.train()

    if loss_mode == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_mode == "nll":
        criterion = None
    else:
        raise ValueError(f"Invalid loss_mode: {loss_mode}")

    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = (criterion(output, target) if loss_mode == "ce"
                else F.nll_loss(output, target))
        loss.backward()
        optimizer.step()

        _record_reads_if_possible(optimizer, data.size(0))

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    train_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Train Epoch: {epoch} 완료 (train acc {train_acc:.2f}%)... ", end="")
    return train_acc


# ---------------------------------------------------------------------------
# Online (batch=1) training — canonical V3.0 pipeline
# ---------------------------------------------------------------------------
def train_online(model, device, train_loader, optimizer, epoch, loss_mode="ce",
                 micro_batch: int = 1, max_samples_per_epoch: int | None = None,
                 progress_every: int = 500):
    """Per-sample (or tiny-micro-batch) training.

    This matches MLP_NeuroSim V3.0's canonical setup: one pulse-update per
    presented sample, instead of one update per averaged mini-batch.

    Notes:
      * BatchNorm layers are forced to eval mode (running stats) because BN
        with batch_size=1 in train mode produces zero-variance NaN gradients.
      * `micro_batch` (default 1) lets you trade fidelity for speed; 4-8 is
        still much closer to "online" than the default 128.
      * `max_samples_per_epoch` caps work per epoch for quick smoke runs.
    """
    model.train()
    bn_count = _set_bn_eval(model)
    if bn_count:
        print(f"[online] forced {bn_count} BatchNorm layer(s) to eval mode")

    correct = 0
    total = 0
    sample_counter = 0
    cap = max_samples_per_epoch

    for data_b, target_b in train_loader:
        data_b = data_b.to(device); target_b = target_b.to(device)
        n_in_batch = data_b.size(0)

        for i in range(0, n_in_batch, micro_batch):
            x = data_b[i:i + micro_batch]
            y = target_b[i:i + micro_batch]

            optimizer.zero_grad()
            output = model(x)
            loss = (nn.CrossEntropyLoss()(output, y) if loss_mode == "ce"
                    else F.nll_loss(output, y))
            loss.backward()
            optimizer.step()
            _record_reads_if_possible(optimizer, x.size(0))

            pred = output.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            sample_counter += x.size(0)

            if progress_every and sample_counter % progress_every == 0:
                print(f"  [online] epoch {epoch}  samples={sample_counter}  "
                      f"running_train_acc={100.0*correct/max(total,1):.2f}%")

            if cap is not None and sample_counter >= cap:
                break
        if cap is not None and sample_counter >= cap:
            break

    train_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Train Epoch (online): {epoch} 완료 (train acc {train_acc:.2f}%)... ", end="")
    return train_acc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def test(model, device, test_loader, loss_mode="ce", return_preds=False):
    """return_preds=True ⇒ (acc, preds, targets)."""
    model.eval()
    correct = 0
    total_loss = 0.0
    all_preds = []
    all_targets = []

    if loss_mode == "ce":
        criterion = nn.CrossEntropyLoss(reduction="sum")
    elif loss_mode == "nll":
        criterion = None
    else:
        raise ValueError(f"Invalid loss_mode: {loss_mode}")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if loss_mode == "ce":
                total_loss += criterion(output, target).item()
            else:
                total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if return_preds:
                all_preds.extend(pred.view(-1).cpu().numpy().tolist())
                all_targets.extend(target.view(-1).cpu().numpy().tolist())

    total_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {total_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")

    if return_preds:
        return acc, all_preds, all_targets
    return acc


def test_specific_letters(model, device, full_test_dataset, target_chars="GACHON"):
    model.eval()
    unique_target_chars = sorted(list(set(target_chars)))
    alphabet_map = {chr(65 + i): i for i in range(26)}
    target_original_labels = [alphabet_map[char] for char in unique_target_chars]

    if "label" not in full_test_dataset.data_frame.columns:
        print("[에러] 테스트 데이터에 'label' 열이 없습니다.")
        return {char: 0.0 for char in unique_target_chars}

    indices = full_test_dataset.data_frame[
        full_test_dataset.data_frame["label"].isin(target_original_labels)
    ].index
    subset = Subset(full_test_dataset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False)

    if len(subset) == 0:
        print(f"[경고] '{target_chars}'에 해당하는 데이터가 테스트셋에 없습니다.")
        return {char: 0.0 for char in unique_target_chars}

    class_correct = {char: 0 for char in unique_target_chars}
    class_total = {char: 0 for char in unique_target_chars}
    adjusted_map = {i: chr(65 + i) for i in range(9)}
    adjusted_map.update({i: chr(65 + i + 1) for i in range(9, 24)})

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                label_idx = labels[i].item()
                pred_idx = predicted[i].item()
                if label_idx in adjusted_map:
                    char_label = adjusted_map[label_idx]
                    if char_label in class_total:
                        class_total[char_label] += 1
                        if label_idx == pred_idx:
                            class_correct[char_label] += 1
    return {
        char: (100.0 * class_correct[char] / class_total[char]) if class_total[char] > 0 else 0.0
        for char in unique_target_chars
    }
