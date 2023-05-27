import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
imagenet_valset = datasets.ImageFolder('C://Users//Adydio//Desktop//try//tiny-imagenet-200//tiny-imagenet-200//val', transform=transform)
model = models.resnet18(pretrained=False)
checkpoint1 = torch.load('checkpoint1.pth')
checkpoint2 = torch.load('checkpoint2.pth')

model.eval()

correct_count1 = 0
correct_count2 = 0
different_images = []

with torch.no_grad():
    for i, (input, target) in enumerate(imagenet_valset):
        model.load_state_dict(checkpoint1['state_dict'])
        output1 = model(input.unsqueeze(0))
        predicted1 = torch.argmax(output1)

        model.load_state_dict(checkpoint2['state_dict'])
        output2 = model(input.unsqueeze(0))
        predicted2 = torch.argmax(output2)

        if predicted1 == target:
            correct_count1 += 1
        if predicted2 == target:
            correct_count2 += 1
        if predicted1 != predicted2:
            different_images.append(i)

print(f"第一次评估准确率: {correct_count1 / len(imagenet_valset)}")
print(f"第二次评估准确率: {correct_count2 / len(imagenet_valset)}")
print(f"评判结果不同的图片索引: {different_images[:10]}")
