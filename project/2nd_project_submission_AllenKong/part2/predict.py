##############USAGE#################
#python predict.py flowers/train/87/image_05467.jpg checkpoint.pth

from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from torchvision import transforms, models
import torch
from torch import nn
import torch.nn.functional as F
from get_input_args import get_input_args_predict
import torch
from data_import import transform_data
print('----------import sucessful----------')

def test(model):
    model.to(device)
    # test network
    accuracy = 0
    with torch.no_grad():
        model.eval()
    # validation pass here
    for images, labels in test_dataloaders:
        images, labels = images.to(device), labels.to(device)
        log_ps = model(images)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    model.train()
    print('test accuracy: {:.3f}%'.format(accuracy/len(test_dataloaders)*100))


in_arg = get_input_args_predict()
if not in_arg.data_dir:
    data_dir = 'flowers'
else:
    data_dir = in_arg.data_dir
print(data_dir)
_, _, _, _, _, test_dataloaders = transform_data(data_dir)
device = torch.device('cuda') if in_arg.gpu else torch.device('cpu')
print('cuda activated:', torch.cuda.is_available())

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    dropout = checkpoint['dropout']
    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    model.classifier = checkpoint['architecture']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict = checkpoint['state_dict']
    return model

checkpoint_path = in_arg.checkpoint
# checkpoint_path = 'checkpoint.pth'
model = load_checkpoint(checkpoint_path)
model.to(device)

print("----------start testing----------")
# test(model)
print("----------done testing----------")

print("----------processing image----------")

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image).transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    if title is not None:
        ax.set_title(title)
        
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def sanity_check(images, model, json_file):

    # Get the actual class labels from the JSON file
    probs, classes = predict(images, model)
    labels = [json_file[label] for label in classes]
    plt.figure(figsize = (10, 4))

    ax= plt.subplot(2,1,1)
    img = process_image(images)
    ax = imshow(img, ax=ax, title=labels[0])
    ax.axis('off')

    plt.subplot(2,1,2)
    sb.barplot(x=probs, y=labels, color=sb.color_palette()[4]);


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    trans = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = Image.open(image)
    processed_img = np.array(trans(img))

    return processed_img

topk=in_arg.topk
print("----------prediction result----------")
# topk=5
def predict(image_path, model, topk=5):
    processed_image = process_image(image_path)
    tensor_image = torch.from_numpy(processed_image).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image = tensor_image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor_image)
    output_probs = torch.exp(output)

    probs, classes = torch.topk(output_probs, topk)
    top_probs = probs.to('cpu').squeeze().tolist()
    top_classes = classes.to('cpu').squeeze().tolist()

    mapping = {value: key for key, value in model.class_to_idx.items()}
    labels = [mapping[idx] for idx in top_classes]

    return top_probs, labels

images = in_arg.input #flowers/train/87/image_05467.jpg
# images = 'flowers/test/62/image_08172.jpg'
top_probs, labels = predict(images, model, topk)
print("Class number:", labels[0])
if in_arg.category_names:
    with open(in_arg.category_names, 'r') as f:
    # with open('/kaggle/input/cat-to-name/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    labels = [cat_to_name[label] for label in labels]
print('probability: {:3f}%..  \nflower name: {}.. '.format(max(top_probs)*100, labels[0]))
# sanity_check(images, model, cat_to_name)