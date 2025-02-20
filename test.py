from datasets import CurriculumNoduleDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def collate_fn(batch):
    return tuple(zip(*batch))

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_dataset = CurriculumNoduleDataset("./refineddataset/trainxrays", "./refineddataset/control", "./refineddataset/nodules.json", "./refineddataset/difficulties.json", 0, transform)
train_dataset.set_difficulty(-3.1)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


print(len(train_loader))

train_loader.dataset.set_difficulty(-0.219)

print(len(train_loader))

print(len(train_loader) * 4, len(train_loader.dataset.idx_is_negative))