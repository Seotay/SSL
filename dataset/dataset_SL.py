import numpy as np
import pandas as pd
import torch
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def load_wm811k_pickles(labeled_path):
    labeled_df = pd.read_pickle(labeled_path)
    
    return labeled_df


class WaferMapToOneHotTensor:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes

    def __call__(self, img):
        arr = np.asarray(img, dtype=np.int64)
        arr = np.clip(arr, 0, self.num_classes - 1)
        tensor = torch.from_numpy(arr)
        one_hot = torch.nn.functional.one_hot(tensor, num_classes=self.num_classes)
        return one_hot.permute(2, 0, 1).to(dtype=torch.float32)



def stratified_split(labels, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1, label_ratio=1.0, data_seed=42):
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
        "train_ratio + val_ratio + test_ratio must be 1.0"
    assert 0 < label_ratio <= 1.0, "label_ratio must be in (0, 1]"

    rng = np.random.RandomState(data_seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))

        train_part = idx[:n_train]
        val_part = idx[n_train:n_train + n_val]
        test_part = idx[n_train + n_val:]

        # n(train data) * label_ratio
        n_labeled = int(np.floor(len(train_part) * label_ratio))
        labeled_train_part = train_part[:n_labeled]

        train_idx.extend(labeled_train_part)
        val_idx.extend(val_part)
        test_idx.extend(test_part)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)



class WM811KSL(Dataset):
    def __init__(self, df, idx=None, transform=None):
        if idx is not None:
            self.df = df.iloc[idx].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.images = self.df["waferMap"].tolist()
        labels = self.df["label"]
        if pd.api.types.is_numeric_dtype(labels):
            self.labels = labels.fillna(-1).to_numpy(dtype=np.int64)
        else:
            self.labels = pd.to_numeric(labels, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_wm811k(labeled_path, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1, label_ratio=0.5, image_size=128, data_seed=42):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        WaferMapToOneHotTensor(),
    ])


    labeled_df = load_wm811k_pickles(labeled_path)
    labels = labeled_df["label"].to_numpy(dtype=np.int64)

    train_labeled_idxs, val_labeled_idxs, test_labeled_idxs = stratified_split(
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        label_ratio=label_ratio,
        data_seed=data_seed,
    )

    train_labeled_dataset = WM811KSL(
        df=labeled_df,
        idx=train_labeled_idxs,
        transform=transform,
    )

    val_dataset = WM811KSL(
        df=labeled_df,
        idx=val_labeled_idxs,
        transform=transform,
    )


    test_dataset = WM811KSL(
        df=labeled_df,
        idx=test_labeled_idxs,
        transform=transform,
    )

    return train_labeled_dataset, val_dataset, test_dataset


def print_dataset_distribution(name, dataset):
    if "label" not in dataset.df.columns:
        print(f"\n[{name}]")
        print("Label column not found.")
        print(f"Total: {len(dataset.df)}")
        return

    counts = dataset.df["label"].value_counts().sort_index()
    total = len(dataset.df)

    idx_to_class = {
        0: "Center",
        1: "Donut",
        2: "Edge-Loc",
        3: "Edge-Ring",
        4: "Loc",
        5: "Random",
        6: "Scratch",
        7: "Near-Full",
        8: "none",
    }

    print(f"\n[{name}]")
    print(f"Total: {total}")

    for i in range(9):
        count = counts.get(i, 0)
        ratio = (count / total * 100) if total > 0 else 0.0
        print(f"{i} ({idx_to_class[i]}): {count} ({ratio:.2f}%)")


if __name__ == "__main__":
    train_labeled_dataset, val_dataset, test_dataset = get_wm811k(
        labeled_path="./data/wm811k/preprocessing/labeled.pkl",
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.1,
        label_ratio=1.0,
        image_size=96,
        data_seed=0,
    )

    print("Train labeled size:", len(train_labeled_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    print_dataset_distribution("Train labeled", train_labeled_dataset)
    print_dataset_distribution("Val", val_dataset)
    print_dataset_distribution("Test", test_dataset)
