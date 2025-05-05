# Ai-end-sem-project
import torch
import torchvision
import numpy as np
import os
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
import random
import copy
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.makedirs('./Dataset/', exist_ok=True)

# PART A: 8-Puzzle Dataset Generation
mnist_train = torchvision.datasets.MNIST(
    root='./Dataset/',
    train=True,
    download=True,
    transform=None
)

mnist_test = torchvision.datasets.MNIST(
    root='./Dataset/',
    train=False,
    download=True,
    transform=None
)

targets = mnist_train.targets.numpy()
classes, class_counts = np.unique(targets, return_counts=True)
num_classes = len(classes)

samples_per_class = 1000
balanced_indices = []

for class_idx in range(num_classes):
    class_indices = np.where(targets == class_idx)[0]
    selected_indices = class_indices[:samples_per_class]
    balanced_indices.extend(selected_indices)

random.shuffle(balanced_indices)
balanced_subset = Subset(mnist_train, balanced_indices)

imbalanced_indices = []
imbal_class_counts = [1000, 1000, 1000, 1000, 1000, 500, 500, 500, 500, 500]

for class_idx in range(num_classes):
    class_indices = np.where(targets == class_idx)[0]
    selected_indices = class_indices[:imbal_class_counts[class_idx]]
    imbalanced_indices.extend(selected_indices)

random.shuffle(imbalanced_indices)
imbalanced_subset = Subset(mnist_train, imbalanced_indices)

torch.save(balanced_subset, './Dataset/mnist_balanced.pt')
torch.save(imbalanced_subset, './Dataset/mnist_imbalanced.pt')

basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

augmented_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Normalize((0.5,), (0.5,))
])

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]

        # Convert tensor to PIL Image if needed before applying transforms
        if isinstance(x, torch.Tensor) and self.transform is not None:
            # Convert tensor to PIL Image
            x = transforms.ToPILImage()(x)

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.subset)

# Add TransformedSubset to safe globals
try:
    torch.serialization.add_safe_globals([TransformedSubset])
except Exception as e:
    print(f"Failed to add safe globals: {e}")

balanced_basic = TransformedSubset(balanced_subset, basic_transform)
balanced_augmented = TransformedSubset(balanced_subset, augmented_transform)
imbalanced_basic = TransformedSubset(imbalanced_subset, basic_transform)
imbalanced_augmented = TransformedSubset(imbalanced_subset, augmented_transform)

torch.save(balanced_basic, './Dataset/mnist_balanced_basic.pt')
torch.save(balanced_augmented, './Dataset/mnist_balanced_augmented.pt')
torch.save(imbalanced_basic, './Dataset/mnist_imbalanced_basic.pt')
torch.save(imbalanced_augmented, './Dataset/mnist_imbalanced_augmented.pt')

class EightPuzzleDataset(Dataset):
    def __init__(self, num_samples=20000, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        self._generate_puzzle_states(num_samples)

    def _generate_puzzle_states(self, num_samples):
        goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

        for _ in range(num_samples):
            current_state = copy.deepcopy(goal_state)
            num_moves = random.randint(5, 50)

            for _ in range(num_moves):
                zero_pos = np.where(current_state == 0)
                x, y = zero_pos[0][0], zero_pos[1][0]

                moves = []
                if x > 0: moves.append((-1, 0))
                if x < 2: moves.append((1, 0))
                if y > 0: moves.append((0, -1))
                if y < 2: moves.append((0, 1))

                dx, dy = random.choice(moves)
                new_x, new_y = x + dx, y + dy

                current_state[x, y], current_state[new_x, new_y] = current_state[new_x, new_y], current_state[x, y]

            manhattan_dist = 0
            for i in range(3):
                for j in range(3):
                    if current_state[i, j] != 0:
                        goal_pos = np.where(goal_state == current_state[i, j])
                        goal_x, goal_y = goal_pos[0][0], goal_pos[1][0]
                        manhattan_dist += abs(i - goal_x) + abs(j - goal_y)

            puzzle_image = np.zeros((28, 28), dtype=np.uint8)
            for i in range(3):
                for j in range(3):
                    start_x, start_y = i*9, j*9
                    value = current_state[i, j]
                    if value != 0:
                        puzzle_image[start_x:start_x+9, start_y:start_y+9] = value * 25

            self.data.append(torch.tensor(puzzle_image, dtype=torch.uint8))
            self.targets.append(manhattan_dist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            # Convert tensor to PIL Image if needed
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, target

# Add EightPuzzleDataset to safe globals
try:
    torch.serialization.add_safe_globals([EightPuzzleDataset])
except Exception as e:
    print(f"Failed to add safe globals: {e}")

puzzle_dataset = EightPuzzleDataset(num_samples=20000)

targets = np.array([t for _, t in puzzle_dataset])
unique_targets, target_counts = np.unique(targets, return_counts=True)

balanced_puzzle_indices = []
samples_per_distance = min(target_counts)

for target_val in unique_targets:
    target_indices = np.where(targets == target_val)[0]
    selected_indices = target_indices[:samples_per_distance]
    balanced_puzzle_indices.extend(selected_indices)

random.shuffle(balanced_puzzle_indices)
balanced_puzzle_subset = Subset(puzzle_dataset, balanced_puzzle_indices)

imbalanced_puzzle_indices = []
for i, target_val in enumerate(unique_targets):
    target_indices = np.where(targets == target_val)[0]
    if i < len(unique_targets) // 2:
        selected_indices = target_indices[:samples_per_distance]
    else:
        selected_indices = target_indices[:samples_per_distance // 2]
    imbalanced_puzzle_indices.extend(selected_indices)

random.shuffle(imbalanced_puzzle_indices)
imbalanced_puzzle_subset = Subset(puzzle_dataset, imbalanced_puzzle_indices)

puzzle_basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

puzzle_augmented_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(15),
    transforms.Normalize((0.5,), (0.5,))
])

balanced_puzzle_basic = TransformedSubset(balanced_puzzle_subset, puzzle_basic_transform)
balanced_puzzle_augmented = TransformedSubset(balanced_puzzle_subset, puzzle_augmented_transform)
imbalanced_puzzle_basic = TransformedSubset(imbalanced_puzzle_subset, puzzle_basic_transform)
imbalanced_puzzle_augmented = TransformedSubset(imbalanced_puzzle_subset, puzzle_augmented_transform)

torch.save(balanced_puzzle_basic, './Dataset/puzzle_balanced_basic.pt')
torch.save(balanced_puzzle_augmented, './Dataset/puzzle_balanced_augmented.pt')
torch.save(imbalanced_puzzle_basic, './Dataset/puzzle_imbalanced_basic.pt')
torch.save(imbalanced_puzzle_augmented, './Dataset/puzzle_imbalanced_augmented.pt')

# PART B: Phone Number Dataset Generation
class PhoneNumberDataset(Dataset):
    def __init__(self, num_samples=20000, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        self.country_codes = {
            'US': '1',
            'UK': '44',
            'India': '91',
            'China': '86',
            'Brazil': '55',
            'Australia': '61',
            'Germany': '49',
            'Japan': '81',
            'Canada': '1',
            'France': '33'
        }
        self._generate_phone_numbers(num_samples)

    def _generate_phone_numbers(self, num_samples):
        for i in range(num_samples):
            phone_number = self._generate_valid_phone_number()
            phone_image = self._create_phone_number_image(phone_number)
            phone_tensor = torch.tensor(np.array(phone_image), dtype=torch.uint8)
            country_idx = i % 10
            self.data.append(phone_tensor)
            self.targets.append(country_idx)

    def _generate_valid_phone_number(self):
        while True:
            digits = [str(random.randint(0, 9)) for _ in range(10)]
            if max([digits.count(d) for d in set(digits)]) <= 4:
                return ''.join(digits)

    def _create_phone_number_image(self, phone_number):
        img = Image.new('L', (280, 28), color=255)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 4), phone_number, fill=0, font=font)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            # Convert tensor to PIL Image if needed
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, target

# Add PhoneNumberDataset to safe globals
try:
    torch.serialization.add_safe_globals([PhoneNumberDataset])
except Exception as e:
    print(f"Failed to add safe globals: {e}")

phone_dataset = PhoneNumberDataset(num_samples=20000)

targets = np.array([t for _, t in phone_dataset])
unique_targets, target_counts = np.unique(targets, return_counts=True)
balanced_indices = []
samples_per_country = min(target_counts)
for target_val in unique_targets:
    target_indices = np.where(targets == target_val)[0]
    selected_indices = target_indices[:samples_per_country]
    balanced_indices.extend(selected_indices)
random.shuffle(balanced_indices)
balanced_phone_subset = Subset(phone_dataset, balanced_indices)

imbalanced_indices = []
digit_patterns = {}
for idx, (img, _) in enumerate(phone_dataset):
    if isinstance(img, torch.Tensor):
        img_np = img.numpy()
    else:
        img_np = np.array(img)
    digit_counts = {}
    for i in range(10):
        digit_counts[str(i)] = 0
    for i in range(10):
        digit = str(i)
        if digit in digit_counts and digit_counts[digit] < 4:
            digit_counts[digit] += 1
    pattern = tuple(sorted([(d, c) for d, c in digit_counts.items() if c >= 3]))
    if pattern:
        if pattern not in digit_patterns:
            digit_patterns[pattern] = []
        digit_patterns[pattern].append(idx)
for pattern, indices in digit_patterns.items():
    imbalanced_indices.extend(indices[:min(len(indices), 50)])
while len(imbalanced_indices) < len(balanced_indices):
    idx = random.randint(0, len(phone_dataset)-1)
    if idx not in imbalanced_indices:
        imbalanced_indices.append(idx)
random.shuffle(imbalanced_indices)
imbalanced_phone_subset = Subset(phone_dataset, imbalanced_indices[:len(balanced_indices)])

phone_basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CountryCodeTransform(Dataset):
    def __init__(self, subset, country_codes, transform=None):
        self.subset = subset
        self.transform = transform
        self.country_codes = country_codes
    def __getitem__(self, idx):
        image, target = self.subset[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        country = list(self.country_codes.keys())[target]
        code = self.country_codes[country]
        new_img = Image.new('L', (image.width + 30, image.height), color=255)
        new_img.paste(image, (30, 0))
        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        draw.text((5, 4), code, fill=0, font=font)
        if self.transform:
            new_img = self.transform(new_img)
        return new_img, target
    def __len__(self):
        return len(self.subset)

# Add CountryCodeTransform to safe globals
try:
    torch.serialization.add_safe_globals([CountryCodeTransform])
except Exception as e:
    print(f"Failed to add safe globals: {e}")

balanced_phone_basic = CountryCodeTransform(
    balanced_phone_subset,
    phone_dataset.country_codes,
    phone_basic_transform
)

imbalanced_phone_basic = CountryCodeTransform(
    imbalanced_phone_subset,
    phone_dataset.country_codes,
    phone_basic_transform
)

torch.save(balanced_phone_basic, './Dataset/phone_balanced_basic.pt')
torch.save(imbalanced_phone_basic, './Dataset/phone_imbalanced_basic.pt')

# PART C: 8-Puzzle Neural Network Implementation
class PuzzleMLP(nn.Module):
    def __init__(self, config):
        super(PuzzleMLP, self).__init__()

        input_size = config.get("input_size", 28 * 28)
        hidden_layers = config.get("hidden_layers", [256, 128])
        output_size = config.get("output_size", 1)
        dropout_rate = config.get("dropout_rate", 0.2)

        layers = []

        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

def get_puzzle_dataloaders(config):
    batch_size = config["batch_size"]
    data_path = config["data_path"]

    # Use weights_only=False to avoid unpickling errors
    balanced_data = torch.load(os.path.join(data_path, "puzzle_balanced_basic.pt"), weights_only=False)
    imbalanced_data = torch.load(os.path.join(data_path, "puzzle_imbalanced_basic.pt"), weights_only=False)

    balanced_size = len(balanced_data)
    imbalanced_size = len(imbalanced_data)

    balanced_train_size = int(0.8 * balanced_size)
    imbalanced_train_size = int(0.8 * imbalanced_size)

    balanced_indices = list(range(balanced_size))
    imbalanced_indices = list(range(imbalanced_size))

    random.shuffle(balanced_indices)
    random.shuffle(imbalanced_indices)

    balanced_train_indices = balanced_indices[:balanced_train_size]
    balanced_test_indices = balanced_indices[balanced_train_size:]

    imbalanced_train_indices = imbalanced_indices[:imbalanced_train_size]
    imbalanced_test_indices = imbalanced_indices[imbalanced_train_size:]

    balanced_train = Subset(balanced_data, balanced_train_indices)
    balanced_test = Subset(balanced_data, balanced_test_indices)

    imbalanced_train = Subset(imbalanced_data, imbalanced_train_indices)
    imbalanced_test = Subset(imbalanced_data, imbalanced_test_indices)

    balanced_train_loader = DataLoader(
        balanced_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 2)
    )

    balanced_test_loader = DataLoader(
        balanced_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 2)
    )

    imbalanced_train_loader = DataLoader(
        imbalanced_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 2)
    )

    imbalanced_test_loader = DataLoader(
        imbalanced_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 2)
    )

    return {
        "balanced_train": balanced_train_loader,
        "balanced_test": balanced_test_loader,
        "imbalanced_train": imbalanced_train_loader,
        "imbalanced_test": imbalanced_test_loader
    }

class PuzzleTrainer:
    def __init__(self, config):
        self.config = config
        self.model = PuzzleMLP(config.get("model", {}))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.learning_rate = config.get("training", {}).get("learning_rate", 0.001)
        self.epochs = config.get("training", {}).get("epochs", 10)

        optimizer_name = config.get("training", {}).get("optimizer", "adam")
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.criterion = nn.MSELoss()

        self.dataloaders = get_puzzle_dataloaders(config)

        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_precision": [],
            "val_precision": [],
            "train_recall": [],
            "val_recall": []
        }

    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device).float()

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                train_preds.extend(outputs.squeeze().detach().cpu().numpy().round())
                train_targets.extend(targets.cpu().numpy())

            train_loss = train_loss / len(train_loader.dataset)

            train_accuracy = accuracy_score(train_targets, train_preds)
            train_precision = precision_score(train_targets, train_preds, average='macro', zero_division=0)
            train_recall = recall_score(train_targets, train_preds, average='macro', zero_division=0)

            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device).float()

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.squeeze(), targets)

                    val_loss += loss.item() * inputs.size(0)

                    val_preds.extend(outputs.squeeze().cpu().numpy().round())
                    val_targets.extend(targets.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)

            val_accuracy = accuracy_score(val_targets, val_preds)
            val_precision = precision_score(val_targets, val_preds, average='macro', zero_division=0)
            val_recall = recall_score(val_targets, val_preds, average='macro', zero_division=0)

            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["train_accuracy"].append(train_accuracy)
            self.metrics["val_accuracy"].append(val_accuracy)
            self.metrics["train_precision"].append(train_precision)
            self.metrics["val_precision"].append(val_precision)
            self.metrics["train_recall"].append(train_recall)
            self.metrics["val_recall"].append(val_recall)

            print(f'Epoch {epoch+1}/{self.epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Train Acc: {train_accuracy:.4f} | '
                  f'Val Acc: {val_accuracy:.4f}')

    def plot_metrics(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_loss"], label="Train Loss")
        plt.plot(self.metrics["val_loss"], label="Validation Loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_accuracy"], label="Train Accuracy")
        plt.plot(self.metrics["val_accuracy"], label="Validation Accuracy")
        plt.title("Accuracy over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "accuracy.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_precision"], label="Train Precision")
        plt.plot(self.metrics["val_precision"], label="Validation Precision")
        plt.title("Precision over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "precision.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_recall"], label="Train Recall")
        plt.plot(self.metrics["val_recall"], label="Validation Recall")
        plt.title("Recall over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "recall.png"))

    def run_experiment(self):
        print("Training on balanced dataset, testing on imbalanced dataset...")
        self.train(
            self.dataloaders["balanced_train"],
            self.dataloaders["imbalanced_test"]
        )
        self.plot_metrics("./results/balanced_to_imbalanced")

        self.metrics = {key: [] for key in self.metrics}
        self.model = PuzzleMLP(self.config.get("model", {}))
        self.model = self.model.to(self.device)

        print("Training on imbalanced dataset, testing on balanced dataset...")
        self.train(
            self.dataloaders["imbalanced_train"],
            self.dataloaders["balanced_test"]
        )
        self.plot_metrics("./results/imbalanced_to_balanced")

# PART D: Phone Number Neural Network Implementation
class PhoneNumberMLP(nn.Module):
    def __init__(self, config):
        super(PhoneNumberMLP, self).__init__()

        # Get the actual input size from a sample
        # For phone numbers, this will be larger than 784
        input_size = config.get("input_size", 28 * 310)  # Updated to match phone number image size
        hidden_layers = config.get("hidden_layers", [256, 128])
        output_size = config.get("output_size", 10)
        dropout_rate = config.get("dropout_rate", 0.2)

        layers = []

        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Reshape input to (batch_size, -1) to flatten it
        x = x.view(x.size(0), -1)
        return self.model(x)

def get_phone_dataloaders(config):
    batch_size = config["batch_size"]
    data_path = config["data_path"]

    # Use weights_only=False to avoid unpickling errors
    balanced_data = torch.load(os.path.join(data_path, "phone_balanced_basic.pt"), weights_only=False)
    imbalanced_data = torch.load(os.path.join(data_path, "phone_imbalanced_basic.pt"), weights_only=False)

    balanced_size = len(balanced_data)
    imbalanced_size = len(imbalanced_data)

    balanced_train_size = int(0.8 * balanced_size)
    imbalanced_train_size = int(0.8 * imbalanced_size)

    balanced_indices = list(range(balanced_size))
    imbalanced_indices = list(range(imbalanced_size))

    random.shuffle(balanced_indices)
    random.shuffle(imbalanced_indices)

    balanced_train_indices = balanced_indices[:balanced_train_size]
    balanced_test_indices = balanced_indices[balanced_train_size:]

    imbalanced_train_indices = imbalanced_indices[:imbalanced_train_size]
    imbalanced_test_indices = imbalanced_indices[imbalanced_train_size:]

    balanced_train = Subset(balanced_data, balanced_train_indices)
    balanced_test = Subset(balanced_data, balanced_test_indices)

    imbalanced_train = Subset(imbalanced_data, imbalanced_train_indices)
    imbalanced_test = Subset(imbalanced_data, imbalanced_test_indices)

    balanced_train_loader = DataLoader(
        balanced_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 2)
    )

    balanced_test_loader = DataLoader(
        balanced_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 2)
    )

    imbalanced_train_loader = DataLoader(
        imbalanced_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 2)
    )

    imbalanced_test_loader = DataLoader(
        imbalanced_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 2)
    )

    return {
        "balanced_train": balanced_train_loader,
        "balanced_test": balanced_test_loader,
        "imbalanced_train": imbalanced_train_loader,
        "imbalanced_test": imbalanced_test_loader
    }

class PhoneNumberTrainer:
    def __init__(self, config):
        self.config = config
        self.model = PhoneNumberMLP(config.get("model", {}))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.learning_rate = config.get("training", {}).get("learning_rate", 0.001)
        self.epochs = config.get("training", {}).get("epochs", 10)

        optimizer_name = config.get("training", {}).get("optimizer", "adam")
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.criterion = nn.CrossEntropyLoss()

        self.dataloaders = get_phone_dataloaders(config)

        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_precision": [],
            "val_precision": [],
            "train_recall": [],
            "val_recall": []
        }

    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())

            train_loss = train_loss / len(train_loader.dataset)

            train_accuracy = accuracy_score(train_targets, train_preds)
            train_precision = precision_score(train_targets, train_preds, average='macro', zero_division=0)
            train_recall = recall_score(train_targets, train_preds, average='macro', zero_division=0)

            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)

            val_accuracy = accuracy_score(val_targets, val_preds)
            val_precision = precision_score(val_targets, val_preds, average='macro', zero_division=0)
            val_recall = recall_score(val_targets, val_preds, average='macro', zero_division=0)

            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["train_accuracy"].append(train_accuracy)
            self.metrics["val_accuracy"].append(val_accuracy)
            self.metrics["train_precision"].append(train_precision)
            self.metrics["val_precision"].append(val_precision)
            self.metrics["train_recall"].append(train_recall)
            self.metrics["val_recall"].append(val_recall)

            print(f'Epoch {epoch+1}/{self.epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Train Acc: {train_accuracy:.4f} | '
                  f'Val Acc: {val_accuracy:.4f}')

    def plot_metrics(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_loss"], label="Train Loss")
        plt.plot(self.metrics["val_loss"], label="Validation Loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_accuracy"], label="Train Accuracy")
        plt.plot(self.metrics["val_accuracy"], label="Validation Accuracy")
        plt.title("Accuracy over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "accuracy.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_precision"], label="Train Precision")
        plt.plot(self.metrics["val_precision"], label="Validation Precision")
        plt.title("Precision over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "precision.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_recall"], label="Train Recall")
        plt.plot(self.metrics["val_recall"], label="Validation Recall")
        plt.title("Recall over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "recall.png"))

    def run_experiment(self):
        print("Training on balanced dataset, testing on imbalanced dataset...")
        self.train(
            self.dataloaders["balanced_train"],
            self.dataloaders["imbalanced_test"]
        )
        self.plot_metrics("./results/phone_balanced_to_imbalanced")

        self.metrics = {key: [] for key in self.metrics}
        self.model = PhoneNumberMLP(self.config.get("model", {}))
        self.model = self.model.to(self.device)

        print("Training on imbalanced dataset, testing on balanced dataset...")
        self.train(
            self.dataloaders["imbalanced_train"],
            self.dataloaders["balanced_test"]
        )
        self.plot_metrics("./results/phone_imbalanced_to_balanced")

# Create config.json
config = {
    "batch_size": 64,
    "data_path": "./Dataset/",
    "num_workers": 2,
    "model": {
        "input_size": 784,  # Default for 8-Puzzle (28x28)
        "hidden_layers": [256, 128, 64],
        "output_size": 1,
        "dropout_rate": 0.2
    },
    "training": {
        "learning_rate": 0.001,
        "epochs": 20,
        "optimizer": "adam"
    }
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

# Main execution
if __name__ == "__main__":
    # Run 8-Puzzle trainer
    puzzle_config = config.copy()
    puzzle_config["model"]["output_size"] = 1  # Regression task
    puzzle_trainer = PuzzleTrainer(puzzle_config)
    puzzle_trainer.run_experiment()

    # Run Phone Number trainer with updated input size
    phone_config = config.copy()
    phone_config["model"]["input_size"] = 28 * 310  # Updated for phone number images (28x310)
    phone_config["model"]["output_size"] = 10  # Classification task with 10 classes
    phone_trainer = PhoneNumberTrainer(phone_config)
    phone_trainer.run_experiment()
