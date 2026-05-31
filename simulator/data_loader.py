# data_loader.py
import pandas as pd
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import config

def read_split_by_write_vh(file_path):
    if not os.path.exists(file_path):
        print(f"[에러] 파일을 찾을 수 없습니다: {file_path}")
        return None, None
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        ltp_df = df[df['WriteVH'] > 0].sort_values(by='PulseNum').reset_index(drop=True)
        ltd_df = df[df['WriteVH'] < 0].sort_values(by='PulseNum').reset_index(drop=True)
        if ltp_df.empty or ltd_df.empty: return None, None
        return ltp_df, ltd_df
    except Exception as e:
        print(f"[에러] 파일 읽기 실패: {e}")
        return None, None

def _balance_classwise(test_dataset, targets_array, num_classes):
    """클래스별 최소 개수에 맞춰 균형 테스트셋 Subset 반환."""
    class_indices = [np.where(targets_array == i)[0] for i in range(num_classes)]
    min_count = min(len(idx) for idx in class_indices)
    print(f"[INFO] 테스트셋 균형 조정: 클래스당 {min_count}개 사용")
    balanced_indices = []
    for i in range(num_classes):
        idx = class_indices[i].copy()
        np.random.shuffle(idx)
        balanced_indices.extend(idx[:min_count])
    return Subset(test_dataset, balanced_indices)


# Light augmentation applied to the *train* stream of any 28x28 grayscale
# dataset when augment=True (matches the ASL augmentation style). Operates on
# PIL images, so it goes before ToTensor.
def _aug_affine():
    return transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1))


def _mnist_like_transforms(mean, std, augment):
    norm = transforms.Normalize((mean,), (std,))
    aug = [_aug_affine()] if augment else []
    train_tf = transforms.Compose(aug + [transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    return train_tf, test_tf


def get_mnist_loaders(batch_size, balanced_test=False, augment=False):
    train_tf, test_tf = _mnist_like_transforms(0.1307, 0.3081, augment)
    train_dataset = datasets.MNIST(config.DATA_DIR, train=True, download=True, transform=train_tf)
    test_dataset = datasets.MNIST(config.DATA_DIR, train=False, download=True, transform=test_tf)

    if balanced_test:
        test_dataset = _balance_classwise(test_dataset, test_dataset.targets.numpy(), 10)
    else:
        print("[INFO] 전체 테스트셋 사용")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_fashion_mnist_loaders(batch_size, balanced_test=False, augment=False):
    train_tf, test_tf = _mnist_like_transforms(0.2860, 0.3530, augment)
    train_dataset = datasets.FashionMNIST(config.DATA_DIR, train=True, download=True, transform=train_tf)
    test_dataset = datasets.FashionMNIST(config.DATA_DIR, train=False, download=True, transform=test_tf)

    if balanced_test:
        test_dataset = _balance_classwise(test_dataset, test_dataset.targets.numpy(), 10)
    else:
        print("[INFO] 전체 테스트셋 사용")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_kmnist_loaders(batch_size, balanced_test=False, augment=False):
    train_tf, test_tf = _mnist_like_transforms(0.1918, 0.3483, augment)
    train_dataset = datasets.KMNIST(config.DATA_DIR, train=True, download=True, transform=train_tf)
    test_dataset = datasets.KMNIST(config.DATA_DIR, train=False, download=True, transform=test_tf)

    if balanced_test:
        test_dataset = _balance_classwise(test_dataset, test_dataset.targets.numpy(), 10)
    else:
        print("[INFO] 전체 테스트셋 사용")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _cifar_transform(augment=False):
    # RGB CIFAR -> grayscale 28x28 (기존 1ch 28x28 모델 호환용)
    pre = [transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28))]
    aug = [_aug_affine()] if augment else []
    return transforms.Compose(
        pre + aug + [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )


def get_cifar10_loaders(batch_size, balanced_test=False, augment=False):
    train_dataset = datasets.CIFAR10(config.DATA_DIR, train=True, download=True,
                                     transform=_cifar_transform(augment))
    test_dataset = datasets.CIFAR10(config.DATA_DIR, train=False, download=True,
                                    transform=_cifar_transform(False))

    if balanced_test:
        test_dataset = _balance_classwise(test_dataset, np.array(test_dataset.targets), 10)
    else:
        print("[INFO] 전체 테스트셋 사용")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_cifar100_loaders(batch_size, balanced_test=False, augment=False):
    train_dataset = datasets.CIFAR100(config.DATA_DIR, train=True, download=True,
                                      transform=_cifar_transform(augment))
    test_dataset = datasets.CIFAR100(config.DATA_DIR, train=False, download=True,
                                     transform=_cifar_transform(False))

    if balanced_test:
        test_dataset = _balance_classwise(test_dataset, np.array(test_dataset.targets), 100)
    else:
        print("[INFO] 전체 테스트셋 사용")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Confusion Matrix / plot 에서 사용할 사람이 읽을 수 있는 클래스 라벨
DATASET_CLASS_NAMES = {
    "MNIST":         [str(i) for i in range(10)],
    "Fashion-MNIST": ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'],
    "K-MNIST":       [str(i) for i in range(10)],   # 일본어 폰트 회피용 인덱스
    "CIFAR-10":      ['plane', 'car', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck'],
    "CIFAR-100":     [str(i) for i in range(100)],  # 100개라 인덱스로 표기
    # ASL(Sign Language MNIST): 0~8 -> A~I, 9~23 -> K~Y (J 없음)
    "ASL":           [chr(65 + i) for i in range(9)] + [chr(65 + i + 1) for i in range(9, 24)],
}


def get_class_names(dataset_name, num_classes=None):
    """데이터셋 이름으로 class label 리스트 반환. 등록되어 있지 않으면 인덱스 사용."""
    if dataset_name in DATASET_CLASS_NAMES:
        return DATASET_CLASS_NAMES[dataset_name]
    if dataset_name.startswith("ASL"):
        return DATASET_CLASS_NAMES["ASL"]
    return [str(i) for i in range(num_classes or 0)]


# 데이터셋 통합 디스패처: name -> (train_loader, test_loader, num_classes)
DATASET_REGISTRY = {
    "MNIST":          {"num_classes": 10,  "loader": get_mnist_loaders,         "supports_balanced": True},
    "Fashion-MNIST":  {"num_classes": 10,  "loader": get_fashion_mnist_loaders, "supports_balanced": True},
    "K-MNIST":        {"num_classes": 10,  "loader": get_kmnist_loaders,        "supports_balanced": True},
    "CIFAR-10":       {"num_classes": 10,  "loader": get_cifar10_loaders,       "supports_balanced": True},
    "CIFAR-100":      {"num_classes": 100, "loader": get_cifar100_loaders,      "supports_balanced": True},
}


def get_loaders(dataset_name, batch_size, balanced_test=False, augment=False):
    """데이터셋 이름으로 (train_loader, test_loader, num_classes) 반환.

    augment=True 면 train 스트림에 가벼운 RandomAffine 증강을 적용한다.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(DATASET_REGISTRY.keys()) + ['ASL']}")
    info = DATASET_REGISTRY[dataset_name]
    train_loader, test_loader = info["loader"](
        batch_size, balanced_test=balanced_test, augment=augment)
    return train_loader, test_loader, info["num_classes"]

class SignMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self): return len(self.data_frame)
    def __getitem__(self, idx):
        original_label = self.data_frame.iloc[idx, 0]
        adjusted_label = original_label - 1 if original_label > 8 else original_label
        label = torch.tensor(adjusted_label, dtype=torch.long)
        pixels = self.data_frame.iloc[idx, 1:].values.astype('uint8').reshape((28, 28))
        image = Image.fromarray(pixels)
        if self.transform: image = self.transform(image)
        return image, label


def get_asl_loaders(batch_size, augment: bool = True):
    """ASL Sign-Language MNIST loaders.

    The official train/test split was collected from *different people* in
    different lighting, so any model that does not augment will overfit the
    train set and plateau around 70~80 % on test (independent of any device
    model). Augmentation (random affine + jitter) is the standard fix in the
    Kaggle leaderboard solutions; see e.g. references to RandAugment-style
    flips, ±10° rotation, small translations, brightness shifts. We apply a
    light version of that here when `augment=True` (default).
    """
    normalise = transforms.Normalize((0.5,), (0.5,))
    if augment:
        train_t = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                    scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        train_t = transforms.Compose([transforms.ToTensor(), normalise])
    test_t = transforms.Compose([transforms.ToTensor(), normalise])

    try:
        train_dataset = SignMNISTDataset(csv_file=config.ASL_TRAIN_PATH, transform=train_t)
        test_dataset  = SignMNISTDataset(csv_file=config.ASL_TEST_PATH,  transform=test_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, test_dataset
    except FileNotFoundError as e:
        print(f"[ERROR] ASL CSV 파일을 찾을 수 없습니다: {e}")
        return None, None, None