import subprocess
import time
import threading
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
from collections import Counter
import torch.cuda.amp as amp
from torch import nn, optim
import copy
from torch.optim import lr_scheduler
from sklearn.model_selection import ParameterSampler

def get_gpu_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                            stdout=subprocess.PIPE)
    return int(result.stdout.decode('utf-8').strip())

class DynamicDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=4, pin_memory=True, prefetch_factor=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.loader = self.create_loader()
        self.adjusting = False
        self.target_gpu_usage = 95

    def create_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory, prefetch_factor=self.prefetch_factor, persistent_workers=True)

    def adjust_num_workers(self):
        while self.adjusting:
            gpu_usage = get_gpu_usage()
            print(f"Current GPU usage: {gpu_usage}%")
            if (gpu_usage < self.target_gpu_usage - 10) and (self.num_workers < 16):
                self.num_workers += 1
                print(f"Increasing num_workers to {self.num_workers}")
            elif (gpu_usage > self.target_gpu_usage + 10) and (self.num_workers > 1):
                self.num_workers -= 1
                print(f"Decreasing num_workers to {self.num_workers}")
            self.loader = self.create_loader()
            time.sleep(20)

    def start_adjusting(self):
        self.adjusting = True
        self.adjust_thread = threading.Thread(target=self.adjust_num_workers)
        self.adjust_thread.start()

    def stop_adjusting(self):
        self.adjusting = False
        self.adjust_thread.join()

    def get_loader(self):
        return self.loader
    
# 파라미터 설정
image_size = 320
num_epochs = 50
ngpu = 1

base_dir = './CatBreeds/'

# 데이터 전처리 및 증강
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 전체 데이터셋 로드
full_dataset = datasets.ImageFolder(base_dir, transform=data_transforms['train'])

# 클래스별 이미지 개수 출력
class_counts = Counter([full_dataset.targets[i] for i in range(len(full_dataset))])
print("Original class distribution:", class_counts)

print("Splitting dataset into training and validation sets...")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

print("Training and validation data are ready.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# EfficientNet-B4 모델 로드
base_model = models.efficientnet_b4(pretrained=True).to(device)

# 모델의 출력 크기를 확인
dummy_input = torch.randn(1, 3, 320, 320).to(device)
base_model.eval()

with torch.no_grad():
    dummy_output = base_model.features(dummy_input)
    num_features = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]
    print(f'Output features: {num_features}')

class CustomModel(nn.Module):
    def __init__(self, base_model, num_classes, dropout):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(num_features, num_classes)
       
    def forward(self, x):
        x = self.base_model.features(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_and_evaluate(params):
    batch_size = params['batch_size']
    lr = params['lr']
    weight_decay = params['weight_decay']
    dropout = params['dropout']

    dynamic_loader = DynamicDataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=4)
    dynamic_loader.start_adjusting()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    dataloaders = {'train': dynamic_loader.get_loader(), 'val': val_loader}

    model = CustomModel(base_model, len(class_names), dropout).to(device)
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = amp.GradScaler()
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataloaders['train']), epochs=num_epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 12
    trigger_times = 0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                trigger_times = 0
            elif phase == 'val':
                trigger_times += 1
                if trigger_times >= patience:
                    model.load_state_dict(best_model_wts)
                    dynamic_loader.stop_adjusting()
                    return best_acc

        scheduler.step()

    model.load_state_dict(best_model_wts)
    dynamic_loader.stop_adjusting()
    return best_acc.item()


# 이 셀은 1시간 30분정도 걸림

param_dist = {
    'batch_size': [16, 32, 64],
    'lr': [0.001, 0.0001, 0.00001],
    'weight_decay': [0.01, 0.001, 0.0001],
    'dropout': [0.3, 0.5, 0.7]
}

param_list = list(ParameterSampler(param_dist, n_iter=5, random_state=42))

best_params = None
best_acc = 0.0

for params in param_list:
    print(f"Evaluating parameters: {params}")
    acc = train_and_evaluate(params)
    print(f"Validation Accuracy: {acc}")
    
    if acc > best_acc:
        best_acc = acc
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best validation accuracy: {best_acc}")

# 최적 하이퍼파라미터로 최종 모델 학습 및 저장
best_batch_size = best_params['batch_size']
best_lr = best_params['lr']
best_weight_decay = best_params['weight_decay']
best_dropout = best_params['dropout']

dynamic_loader = DynamicDataLoader(train_dataset, batch_size=best_batch_size, num_workers=4, pin_memory=True, prefetch_factor=4)
dynamic_loader.start_adjusting()
val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)
dataloaders = {'train': dynamic_loader.get_loader(), 'val': val_loader}

model = CustomModel(base_model, len(class_names), best_dropout).to(device)
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
scaler = amp.GradScaler()
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=best_lr, steps_per_epoch=len(dataloaders['train']), epochs=num_epochs)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
patience = 12
trigger_times = 0

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                with amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger_times = 0
        elif phase == 'val':
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                model.load_state_dict(best_model_wts)
                dynamic_loader.stop_adjusting()
                exit()

        if phase == 'val':
            val_loss = epoch_loss

    scheduler.step()

print('Training complete')
print(f'Best val Acc: {best_acc:4f}')

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'cat_breeds_efficientnet_b4.pth')

# 동적 조정 멈춤
dynamic_loader.stop_adjusting()