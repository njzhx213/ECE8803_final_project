import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

class OLIVESDataset(Dataset):
    def __init__(self, labels_file, transform=None):
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform
        self.path_column = self.labels_df.columns[0]  # First column is the image path
        self.image_paths = self.labels_df[self.path_column].tolist()  # Extract paths
        self.labels_df = self.labels_df.drop(columns=[self.path_column])  # Drop the path column from labels
        self.labels_df = self.labels_df.astype(float)  # Ensure remaining columns are float type
        self.label_columns = self.labels_df.columns.tolist()  # Remaining columns are labels

        print(f"Found path column: {self.path_column}")
        print(f"Found {len(self.label_columns)} label columns")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # Get the image path from the list

        try:
            # Open image and ensure it's grayscale
            image = Image.open("."+img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            # Convert labels to float32
            labels = self.labels_df.iloc[idx].values
            labels = torch.tensor(labels, dtype=torch.float32)

            return image, labels

        except FileNotFoundError as e:
            print(f"Skipping index {idx} due to missing file: {img_path}")
        except TypeError as e:
            print(f"Skipping index {idx} due to label type error: {str(e)}")
        except Exception as e:
            print(f"Skipping index {idx} due to error: {str(e)}")

        # Return a zeroed image and labels if an error occurs
        image = torch.zeros((1, 224, 224), dtype=torch.float32)  # Changed to 1 channel for grayscale
        labels = torch.zeros(len(self.label_columns), dtype=torch.float32)
        return image, labels

    def get_num_classes(self):
        return len(self.label_columns)

# Data Transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Updated for grayscale
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Updated for grayscale
])

def load_data(batch_size, labels_file, is_training=True, num_workers=4, pin_memory=True):
    dataset = OLIVESDataset(
        labels_file=labels_file,
        transform=data_transforms if is_training else val_transforms
    )

    indices = list(range(len(dataset)))
    if is_training:
        np.random.shuffle(indices)
    split = int(len(dataset) * 0.8)

    if is_training:
        sampler = SubsetRandomSampler(indices[:split])
    else:
        sampler = SubsetRandomSampler(indices[split:])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last batch if it is incomplete to avoid any inconsistencies
    )
    return dataloader, dataset.get_num_classes()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            # Keep everything in float32 for evaluation
            images = images.to(device).float()  # Changed from half to float
            labels = labels.to(device).float()

            # Temporarily convert model to float for inference
            model.float()
            outputs = model(images)
            model.half()  # Convert back to half precision

            predictions = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    return metrics

class BiomarkerDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(BiomarkerDetectionModel, self).__init__()

        # Initialize ViT
        self.vit = models.vision_transformer.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()

        # Initialize EfficientNet
        self.efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        eff_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # Projection layers
        self.vit_proj = nn.Linear(num_features, 512)
        self.eff_proj = nn.Linear(eff_features, 512)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        vit_features = self.vit(x)
        eff_features = self.efficientnet(x)

        vit_features = self.vit_proj(vit_features)
        eff_features = self.eff_proj(eff_features)

        vit_features = vit_features.unsqueeze(0)
        eff_features = eff_features.unsqueeze(0)

        attended_features, _ = self.attention(vit_features, eff_features, eff_features)
        features = attended_features.squeeze(0)

        output = self.classifier(features)
        return output

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("WARNING: Using CPU. Training will be very slow!")
    print(f"Using device: {device}")

    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True

    # Paths
    labels_file = "./OLIVES_Dataset_Labels/full_labels/Biomarker_Clinical_Data_Images.csv"
    os.makedirs('model_checkpoints', exist_ok=True)

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.00001  # Reduced learning rate
    num_epochs = 10

    print("Loading datasets...")
    train_dataloader, num_classes = load_data(
        batch_size=batch_size,
        labels_file=labels_file,
        is_training=True,
        num_workers=4,  # Reduced workers
        pin_memory=True
    )

    val_dataloader, _ = load_data(
        batch_size=batch_size,
        labels_file=labels_file,
        is_training=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Number of classes: {num_classes}")

    # Initialize model
    print("Initializing model...")
    model = BiomarkerDetectionModel(num_classes).to(device)

    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Use float32 for training stability
    model = model.float()

    # Loss and optimizer with gradient clipping
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # Add weight decay
        eps=1e-8  # Increase epsilon for stability
    )

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Training loop
    print("Starting training...")
    best_f1 = 0.0

    scaler = torch.amp.GradScaler()  # Updated for mixed precision training

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs}')

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device).float()  # Ensure float32
            labels = labels.to(device).float()

            # Mixed precision training
            #with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            try:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            except ValueError as e:
                print(f"Warning: Skipping batch due to error - {e}")
                continue

            if not torch.isfinite(loss):
                print(f"WARNING: Non-finite loss, skipping batch")
                continue

            running_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        pbar.close()

        if running_loss == 0:
            print("WARNING: Zero loss for entire epoch!")
            continue

        epoch_loss = running_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} average loss: {epoch_loss:.4f}")

        # Validation
        print("Evaluating on validation set...")
        metrics = evaluate_model(model, val_dataloader, device)
        print("Validation Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        scheduler.step(epoch_loss)

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            checkpoint_path = os.path.join('model_checkpoints', f'best_model_f1_{best_f1:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)
            print(f"Saved new best model with F1 score: {best_f1:.4f}")

    print("Training complete!")
    print(f"Best F1 Score achieved: {best_f1:.4f}")

if __name__ == "__main__":
    main()