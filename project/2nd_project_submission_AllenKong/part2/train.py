##############USAGE#################
# python train.py flowers --save_dir ImageClassifier
# python train.py flowers --arch densenet121

#Check torch version and CUDA status if GPU is enabled.
import time
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
from torch import optim
from get_input_args import get_input_args_train
from data_import import transform_data
print('----------import sucessful----------')
in_arg = get_input_args_train()

print('---------loading data----------')
data_dir = in_arg.data_dir
train_image_datasets, valid_image_datasets, test_image_datasets, train_dataloaders, valid_dataloaders, test_dataloaders = transform_data(data_dir)

device = torch.device('cuda') if in_arg.gpu else torch.device('cpu')
print('cuda activated:', torch.cuda.is_available())

input_size = 512*7*7
hidden_size = [int(unit) for unit in in_arg.hidden_units.split(',')]
output_size = 102
dropout = 0.5
lr = in_arg.learning_rate
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
    
arch = in_arg.arch
def setup(input_size, hidden_size, output_size, dropout):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    for param in model.features.parameters():
        param.requires_grad = False

    classifier = Classifier(input_size, hidden_size, output_size, dropout)
    model.classifier = classifier
    
    return model

model = setup(input_size, hidden_size, output_size, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

model.to(device)
print('----------model activated----------')
print('arch {}.. hidden {}.. lr {}.. epochs {}..'.format(arch, hidden_size, lr, in_arg.epochs))
optimizer.zero_grad()
print('----------zero grad sucessfully----------')

print('----------model train starting----------')
train_losses = []
test_losses = []
epochs = in_arg.epochs
for e in range(epochs):
    start = time.time()
    running_loss = 0
    for images, labels in train_dataloaders:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
        # validation pass here
        for images, labels in valid_dataloaders:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            valid_loss += criterion(log_ps, labels).item()
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()

        train_losses.append(running_loss/len(train_dataloaders))
        test_losses.append(valid_loss/len(valid_dataloaders))  
        

        print('Epoch: {}/{}..'.format(e+1, epochs))
        print('training loss: {:.3f}'.format(running_loss/len(train_dataloaders)))
        print('test loss: {:.3f}'.format(valid_loss/len(valid_dataloaders)))
        print('test accuracy: {:.3f}'.format(accuracy/len(valid_dataloaders)))
        print(f'time used, {(time.time() - start)/3:.3f} seconds')
print('----------model train sucessfully----------')

model.class_to_idx = train_image_datasets.class_to_idx

checkpoint = {'dropout': dropout,
              'criterion': criterion,
              'optimizer': optimizer,
              'architecture': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
# checkpoint
print(in_arg.save_dir+'# /checkpoint.pth')
torch.save(checkpoint, in_arg.save_dir+'# /checkpoint.pth')
print('----------checkpoint saved sucessfully----------')