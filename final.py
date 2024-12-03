import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import pandas as pd
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report

# Dataset Class for OLIVES
class UpdatedOLIVESDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        """
        Args:
            annotations (DataFrame): DataFrame with image paths and labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Extract relative image path and remove leading '/' if present
        relative_path = self.img_labels.iloc[idx, 0].lstrip("/")
        img_path = os.path.join(self.img_dir, relative_path)

        # Load the image
        try:
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")  # Convert to RGB for consistency
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify or open the image file: {img_path}")

        # Extract local and global labels using column names
        local_labels = torch.tensor(
            self.img_labels.loc[idx, ["IR_HRF", "IRF", "DRT/ME"]].astype(float).values, dtype=torch.float32
        )
        global_labels = torch.tensor(
            self.img_labels.loc[idx, ["PAVF", "FAVF", "VD"]].astype(float).values, dtype=torch.float32
        )
        labels = torch.cat([local_labels, global_labels], dim=0)

        if self.transform:
            image = self.transform(image)
        return image, labels


# Data Transforms
data_transforms = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to prepare k-fold cross-validation datasets
def prepare_kfold_data(k, annotations, img_dir, transform, batch_size):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    datasets = []
    for train_idx, val_idx in kf.split(annotations):
        train_data = UpdatedOLIVESDataset(
            annotations.iloc[train_idx].reset_index(drop=True),  # Reset index
            img_dir,
            transform=transform,
        )
        val_data = UpdatedOLIVESDataset(
            annotations.iloc[val_idx].reset_index(drop=True),  # Reset index
            img_dir,
            transform=transform,
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
        datasets.append((train_loader, val_loader))
    return datasets

# Define the Model
class BiomarkerDetectionModel(nn.Module):
    def __init__(self):
        super(BiomarkerDetectionModel, self).__init__()
        # Local feature extractor (MaxViT)
        self.local_extractor = models.vision_transformer.vit_b_16(pretrained=True)
        # Access the final linear layer of the heads
        in_features_local = self.local_extractor.heads[-1].in_features
        self.local_extractor.heads = nn.Linear(in_features_local, 3)  # 3 local features

        # Global feature extractor (EVA-02)
        self.global_extractor = models.efficientnet_v2_s(pretrained=True)
        # Access the final linear layer of the classifier
        in_features_global = self.global_extractor.classifier[-1].in_features
        self.global_extractor.classifier[-1] = nn.Linear(in_features_global, 3)  # 3 global features

    def forward(self, x):
        # Extract local features
        local_features = self.local_extractor(x)
        
        # Extract global features
        global_features = self.global_extractor(x)
        
        # Concatenate features
        output = torch.cat([local_features, global_features], dim=1)
        return output

# Evaluation Function
# Evaluation Function with Detailed Metrics
def evaluate_model_detailed(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5  # Threshold for multi-label classification
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Generate a classification report for detailed metrics
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["IR_HRF", "IRF", "DRT/ME", "PAVF", "FAVF", "VD"],
        output_dict=True,
        zero_division=0
    )

    # Extract per-class and overall metrics
    table = "\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Support} \\\\ \\midrule\n"
    for label, metrics in report.items():
        if label.isdigit():  # Check if the key is a class index
            class_name = f"B{int(label) + 1}"
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1-score"]
            support = metrics["support"]
            table += f"{class_name} & {precision:.2f} & {recall:.2f} & {f1:.2f} & {int(support)} \\\\\n"
    
    # Add overall metrics
    table += "\\midrule\n"
    overall_metrics = {
        "Micro Avg": report["micro avg"],
        "Macro Avg": report["macro avg"],
        "Weighted Avg": report["weighted avg"],
        "Samples Avg": report["samples avg"]
    }
    for name, metrics in overall_metrics.items():
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1-score"]
        support = metrics["support"]
        table += f"\\textbf{{{name}}} & {precision:.2f} & {recall:.2f} & {f1:.2f} & {int(support)} \\\\\n"
    
    table += "\\bottomrule"
    print(table)

# Main Function
if __name__ == "__main__":
    # Paths to your dataset files
    annotations_file_path = "OLIVES_Dataset_Labels/full_labels/Biomarker_Clinical_Data_Images.csv"
    img_dir = "OLIVES"  # Root directory containing all images

    # Load Annotations
    annotations_df = pd.read_csv(annotations_file_path)
    
    # Ensure label columns are numeric
    annotations_df.iloc[:, 2:] = annotations_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaN values with 0
    annotations_df.fillna(0, inplace=True)
    
   
    # Filter annotations for necessary columns
    filtered_annotations = annotations_df.loc[:, ['Path', 'IR_HRF', 'IRF', 'DRT/ME', 'PAVF', 'FAVF', 'VD']]

    # Ensure label columns are numeric
    filtered_annotations[['IR_HRF', 'IRF', 'DRT/ME', 'PAVF', 'FAVF', 'VD']] = filtered_annotations[
        ['IR_HRF', 'IRF', 'DRT/ME', 'PAVF', 'FAVF', 'VD']
    ].apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with 0 (or another appropriate default)
    filtered_annotations.fillna(0, inplace=True)
    
    # Print DataFrame structure and first few rows to verify
    print("DataFrame Column Types:")
    print(filtered_annotations.dtypes)
    print("\nFirst 5 Rows of the DataFrame:")
    print(filtered_annotations.head())
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 1
    k_folds = 5

    # Prepare K-Fold Data
    datasets = prepare_kfold_data(k_folds, filtered_annotations, img_dir, data_transforms, batch_size)

    # Initialize Model
    model = BiomarkerDetectionModel().cuda()

    # Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()  # Multi-label loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR every 5 epochs

    # K-Fold Training
    for fold, (train_dataloader, val_dataloader) in enumerate(datasets):
        print(f"Fold {fold + 1}/{k_folds}")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_dataloader:
                images, labels = images.cuda(), labels.cuda()
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Adjust learning rate after each epoch
                scheduler.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")
        
        # Evaluate the Model
        print(f"Evaluating Fold {fold + 1}")
        evaluate_model_detailed(model, val_dataloader)
