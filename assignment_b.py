import copy
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Hyper Parameters
batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 20
num_epochs_pretrain = 5


class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test
        self.mode = mode

        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.get_item(index)
        else:
            return self.get_item_dual(index)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['img_path'])
            if not self.test:
                file_info['dr_level'] = int(row['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    # 2. dual image
    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]  # The patient id of each image
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]  # The left or right eye
        grouped = df.groupby(['prefix', 'suffix'])

        data = []
        for (prefix, suffix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item_dual(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]


class APTOSDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test
        self.mode = mode

        assert mode == 'single', 'only single mode is supported for APTOS data'
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.get_item(index)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['id_code'] + '.png')
            if not self.test:
                file_info['dr_level'] = int(row['diagnosis'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img


class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input image must be a torch.Tensor')

        # Get height and width of the image
        h, w = img.shape[1], img.shape[2]
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        cx = np.random.randint(mask_size_half, w + offset - mask_size_half)
        cy = np.random.randint(mask_size_half, h + offset - mask_size_half)

        xmin, xmax = cx - mask_size_half, cx + mask_size_half + offset
        ymin, ymax = cy - mask_size_half, cy + mask_size_half + offset
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        img[:, ymin:ymax, xmin:xmax] = 0
        return img


class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img


transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((210, 210)),
    SLORandomPad((224, 224)),
    FundRandomRotate(prob=0.5, degree=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.1, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        all_preds = []
        all_labels = []

        model.train()

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                if not isinstance(images, list):
                    images = images.to(device)  # single image case
                else:
                    images = [x.to(device) for x in images]  # dual images case

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())

                pbar.set_postfix({'lr': f'{optimizer.param_groups[0]["lr"]:.1e}', 'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        lr_scheduler.step()

        epoch_loss = sum(running_loss) / len(running_loss)

        train_metrics = compute_metrics(all_preds, all_labels, per_class=True)
        kappa, accuracy, precision, recall = train_metrics[:4]

        print(f'[Train] Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} Loss: {epoch_loss:.4f}')

        if len(train_metrics) > 4:
            precision_per_class, recall_per_class = train_metrics[4:]
            for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
                print(f'[Train] Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}')

        # Evaluation on the validation set at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')

    return model


def evaluate_model(model, test_loader, device, test_only=False, prediction_path='./test_predictions.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

            if test_only:
                images = data
            else:
                images, labels = data

            if not isinstance(images, list):
                images = images.to(device)  # single image case
            else:
                images = [x.to(device) for x in images]  # dual images case

            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

            if not isinstance(images, list):
                # single image case
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in
                    range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.numpy())
            else:
                # dual images case
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(test_loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(i * test_loader.batch_size, i * test_loader.batch_size + len(images[k]))
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.numpy())

            pbar.update(1)

    # Save predictions to csv file for Kaggle online evaluation
    if test_only:
        df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        df.to_csv(prediction_path, index=False)
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
    else:
        metrics = compute_metrics(all_preds, all_labels)
        return metrics


def compute_metrics(preds, labels, per_class=False):
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Calculate and print precision and recall for each class
    if per_class:
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        return kappa, accuracy, precision, recall, precision_per_class, recall_per_class

    return kappa, accuracy, precision, recall

def get_model_predictions_boosting(models, data_loader, device):
    model_preds = []
    model_labels = []
    
    for model in models:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        model_preds.append(all_preds)
        model_labels = all_labels  # Keep the labels for final evaluation

    return np.array(model_preds).T, model_labels  # Model predictions as features

def boosting_ensemble(models, train_loader, device):
    # Get predictions from the base models
    train_preds, train_labels = get_model_predictions_boosting(models, train_loader, device)
    
    # Use AdaBoost with decision trees as the base learner
    base_model = DecisionTreeClassifier(max_depth=1)
    boosting_model = AdaBoostClassifier(base_model, n_estimators=50)
    
    # Train the boosting model
    boosting_model.fit(train_preds, train_labels)
    
    # Evaluate the boosting model
    val_preds = boosting_model.predict(train_preds)  # Using validation data
    print(f'Boosting Validation Accuracy: {accuracy_score(train_labels, val_preds)}')

    return boosting_model

# Step 1: Generate predictions from each model
def get_model_predictions(models, data_loader, device):
    model_preds = []
    model_labels = []
    
    for model in models:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        model_preds.append(all_preds)
        model_labels = all_labels  # Keep the labels for final evaluation

    return np.array(model_preds).T, model_labels  # Model predictions as features

# Step 2: Train a meta-model
def stack_models(models, train_loader, device):
    # Get base model predictions for training data
    train_preds, train_labels = get_model_predictions(models, train_loader, device)
    
    # Split training data for meta-model
    X_train, X_val, y_train, y_val = train_test_split(train_preds, train_labels, test_size=0.2, random_state=42)
    
    # Train the meta-model
    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_train)
    
    # Evaluate the meta-model
    val_preds = meta_model.predict(X_val)
    print(f'Validation Accuracy: {accuracy_score(y_val, val_preds)}')

    return meta_model


def bagging_ensemble(models, train_loader, device):
    # Combine classifiers into a bagging ensemble
    bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
    
    # Get predictions and train the ensemble
    train_preds, train_labels = get_model_predictions_boosting(models, train_loader, device)
    bagging_model.fit(train_preds, train_labels)
    
    # Evaluate the ensemble
    val_preds = bagging_model.predict(train_preds)
    print(f'Bagging Validation Accuracy: {accuracy_score(train_labels, val_preds)}')

    return bagging_model


class ResNet18(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.vgg16(pretrained=True)
        in_features = self.backbone.classifier[0].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = torch.flatten(x, 1)  # flatten the output of the adaptive pooling layer
        x = self.backbone.classifier(x)
        return x
    

class Densenet121(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(Densenet121, self).__init__()
        # Load the pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=True)
        
        # Extract the number of input features for the original classifier
        in_features = self.backbone.classifier.in_features
        
        # Replace the classifier with a custom classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Forward pass through the DenseNet backbone
        x = self.backbone(x)
        return x
    

def load_or_train_model(model, train_loader, val_loader, device, *args, checkpoint_path, force_train=False, **kwargs):
    if not force_train and os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        return model.to(device)

    return train_model(model, train_loader, val_loader, device, *args, checkpoint_path=checkpoint_path, **kwargs)



if __name__ == '__main__':
    # Choose between 'single image' and 'dual images' pipeline
    # This will affect the model definition, dataset pipeline, training and evaluation

    mode = 'single'  # forward single image to the model each time

    custom_pretrain = True # Use our own pretrained models from task b
    custom_pretrain_force_train = False # Retrain pre-trained models even if they exist

    print('Pipeline Mode:', mode)

    model1 = VGG16()
    model2 = Densenet121()
    model3 = ResNet18()

    # Create datasets
    train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train, mode)
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, mode)
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, mode, test=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Use GPU device is possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)


    if custom_pretrain:
        # Pre-train the models from APTOS data

        aptos_train_dataset = APTOSDataset('./APTOS/train_1.csv', './APTOS/train_images/', transform_train, mode)
        aptos_val_dataset = APTOSDataset('./APTOS/valid.csv', './APTOS/val_images/', transform_test, mode)

        aptos_train_loader = DataLoader(aptos_train_dataset, batch_size=batch_size, shuffle=True)
        aptos_val_loader = DataLoader(aptos_val_dataset, batch_size=batch_size, shuffle=False)

        vgg_model = model1.to(device)
        densenet121_model = model2.to(device)
        resnet18_model = model3.to(device)

        optimizer_vgg = torch.optim.Adam(params=model1.parameters(), lr=learning_rate)
        optimizer_densenet121 = torch.optim.Adam(params=model2.parameters(), lr=learning_rate)
        optimizer_resnet18 = torch.optim.Adam(params=model3.parameters(), lr=learning_rate)

        lr_scheduler_vgg = torch.optim.lr_scheduler.StepLR(optimizer_vgg, step_size=10, gamma=0.1)
        lr_scheduler_densenet121 = torch.optim.lr_scheduler.StepLR(optimizer_densenet121, step_size=10, gamma=0.1)
        lr_scheduler_resnet18 = torch.optim.lr_scheduler.StepLR(optimizer_resnet18, step_size=10, gamma=0.1)

        print("Pre-train VGG")
        vgg_model = load_or_train_model(
            vgg_model, aptos_train_loader, aptos_val_loader, device, criterion, optimizer_vgg,
            lr_scheduler=lr_scheduler_vgg, num_epochs=num_epochs_pretrain,
            checkpoint_path='./vgg16_pretrain.pth', force_train=custom_pretrain_force_train
        )
        
        print("Pre-train DenseNet")
        densenet121_model = load_or_train_model(
            densenet121_model, aptos_train_loader, aptos_val_loader, device, criterion, optimizer_densenet121,
            lr_scheduler=lr_scheduler_densenet121, num_epochs=num_epochs_pretrain,
            checkpoint_path='./densenet121_pretrain.pth', force_train=custom_pretrain_force_train
        )
        
        print("Pre-train ResNet")
        resnet18_model = load_or_train_model(
            resnet18_model, aptos_train_loader, aptos_val_loader, device, criterion, optimizer_resnet18,
            lr_scheduler=lr_scheduler_resnet18, num_epochs=num_epochs_pretrain,
            checkpoint_path='./resnet18_pretrain.pth', force_train=custom_pretrain_force_train
        )
    else:
        # Load the provided pretrained models

        vgg16_state_dict = torch.load('./pretrained_DR_resize/pretrained/vgg16.pth', map_location='cpu')
        densenet121_state_dict = torch.load('./pretrained_DR_resize/pretrained/densenet121.pth', map_location='cpu')
        resnet18_state_dict = torch.load('./pretrained_DR_resize/pretrained/resnet18.pth', map_location='cpu')

        model1.load_state_dict(vgg16_state_dict, strict=False)
        model2.load_state_dict(densenet121_state_dict, strict=False)
        model3.load_state_dict(resnet18_state_dict, strict=False)

        vgg_model = model1.to(device)
        densenet121_model = model2.to(device)
        resnet18_model = model3.to(device)

    # Optimizer and Learning rate scheduler
    optimizer_vgg = torch.optim.Adam(params=model1.parameters(), lr=learning_rate)
    optimizer_densenet121 = torch.optim.Adam(params=model2.parameters(), lr=learning_rate)
    optimizer_resnet18 = torch.optim.Adam(params=model3.parameters(), lr=learning_rate)

    lr_scheduler_vgg = torch.optim.lr_scheduler.StepLR(optimizer_vgg, step_size=10, gamma=0.1)
    lr_scheduler_densenet121 = torch.optim.lr_scheduler.StepLR(optimizer_densenet121, step_size=10, gamma=0.1)
    lr_scheduler_resnet18 = torch.optim.lr_scheduler.StepLR(optimizer_resnet18, step_size=10, gamma=0.1)

    vgg_model = train_model(
        vgg_model, train_loader, val_loader, device, criterion, optimizer_vgg,
        lr_scheduler=lr_scheduler_vgg, num_epochs=num_epochs,
        checkpoint_path='./vgg16_transfer.pth'
    )
    
    densenet121_model = train_model(
        densenet121_model, train_loader, val_loader, device, criterion, optimizer_densenet121,
        lr_scheduler=lr_scheduler_densenet121, num_epochs=num_epochs,
        checkpoint_path='./densenet121_transfer.pth'
    )
    
    resnet18_model = train_model(
        resnet18_model, train_loader, val_loader, device, criterion, optimizer_resnet18,
        lr_scheduler=lr_scheduler_resnet18, num_epochs=num_epochs,
        checkpoint_path='./resnet18_transfer.pth'
    )

    evaluate_model(vgg_model, test_loader, device, test_only=True, prediction_path='vgg_pred.csv')
    evaluate_model(densenet121_model, test_loader, device, test_only=True, prediction_path='densenet121_pred.csv')
    evaluate_model(resnet18_model, test_loader, device, test_only=True, prediction_path='resnet18_pred.csv')
