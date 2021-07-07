import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def imshow(inp, title):
    """Imshow for Tensor."""
    import numpy as np
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

def plot(data_dir, val_loss,train_loss,typ):
    plt.title("{} after epoch: {}".format(typ,len(train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel(typ)
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train "+typ)
    plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Validation "+typ)
    plt.legend()
    plt.savefig(os.path.join(data_dir,typ+".png"))
    plt.close()

def train_model(model, criterion, optimizer, scheduler, num_epochs=100,model_name = "kaggle"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) #was (outputs,1) for non-inception and (outputs.data,1) for inception
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
              train_loss_gph.append(epoch_loss)
              train_acc_gph.append(epoch_acc)
            if phase == 'val':
              val_loss_gph.append(epoch_loss)
              val_acc_gph.append(epoch_acc)
            
            plot(data_dir, val_loss_gph, train_loss_gph, typ = "Loss "+model_name)
            plot(data_dir, val_acc_gph, train_acc_gph, typ = "Accuracy "+model_name)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, data_dir+"/"+model_name+".h5")
                print('==>Model Saved')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_resnet34(path,
                   batch_size = 4,
                   num_workers = 2,
                   epochs = 100,
                   learning_rate = 0.001,
                   momentum = 0.99,
                   ):
    print("\nTraining ResNet34")
    import numpy as np
    global mean, std, data_dir
    data_dir = path
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    global dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']}
    global dataset_sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(class_names)

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    global val_loss_gph, train_loss_gph, val_acc_gph, train_acc_gph
    val_loss_gph=[]
    train_loss_gph=[]
    val_acc_gph=[]
    train_acc_gph=[]

    model = models.resnet34(pretrained = True)

    num_ftrs = model.fc.in_features
    print("Number of features: "+str(num_ftrs))
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum)

    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=epochs, model_name = "resnet34")

    # Getting Probability distribution
    print("\nGetting the Probability Distribution")
    testloader=torch.utils.data.DataLoader(image_datasets['val'],batch_size=1)
    model=model.eval()

    correct = 0
    total = 0
    import csv
    import numpy as np
    dst = data_dir+'/csv'
    if not os.path.exists(dst):
        os.makedirs(dst)
    f = open(dst+"/resnet34.csv",'w+',newline = '')
    writer = csv.writer(f)

    saving = []
    with torch.no_grad():
          num = 0
          temp_array = np.zeros((len(testloader),num_classes))
          for i,data in enumerate(testloader):
              images, labels = data
              sample_fname, _ = testloader.dataset.samples[i]
              labels=labels.cuda()
              outputs = model(images.cuda())
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (predicted == labels.cuda()).sum().item()
              prob = torch.nn.functional.softmax(outputs, dim=1)
              saving.append(sample_fname.split('/')[-1])
              temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
              num+=1
    print("Accuracy = ",100*correct/total)

    for i in range(len(testloader)):
      k = temp_array[i].tolist()
      k.append(saving[i])
      writer.writerow(k)

    f.close()

def train_densenet201(path,
                      batch_size = 4,
                      num_workers = 2,
                      epochs = 100,
                      learning_rate = 0.001,
                      momentum = 0.99,
                      ):
    print("\nTraining DenseNet201")
    import numpy as np
    global mean, std, data_dir
    data_dir = path
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    global dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']}
    global dataset_sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(class_names)

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    global val_loss_gph, train_loss_gph, val_acc_gph, train_acc_gph
    val_loss_gph=[]
    train_loss_gph=[]
    val_acc_gph=[]
    train_acc_gph=[]

    model = models.densenet201(pretrained = True)
    
    num_ftrs = model.classifier.in_features
    
    print("Number of features: "+str(num_ftrs))
    
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum)

    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=epochs, model_name = "densenet201")

    # Getting Probability distribution
    print("\nGetting the Probability Distribution")
    testloader=torch.utils.data.DataLoader(image_datasets['val'],batch_size=1)
    model=model.eval()

    correct = 0
    total = 0
    import csv
    import numpy as np
    dst = data_dir+'/csv'
    if not os.path.exists(dst):
        os.makedirs(dst)
    f = open(dst+"/densenet201.csv",'w+',newline = '')
    writer = csv.writer(f)

    saving = []
    with torch.no_grad():
          num = 0
          temp_array = np.zeros((len(testloader),num_classes))
          for i,data in enumerate(testloader):
              images, labels = data
              sample_fname, _ = testloader.dataset.samples[i]
              labels=labels.cuda()
              outputs = model(images.cuda())
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (predicted == labels.cuda()).sum().item()
              prob = torch.nn.functional.softmax(outputs, dim=1)
              saving.append(sample_fname.split('/')[-1])
              temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
              num+=1
    print("Accuracy = ",100*correct/total)

    for i in range(len(testloader)):
      k = temp_array[i].tolist()
      k.append(saving[i])
      writer.writerow(k)

    f.close()

def train_inceptionv3(path,
                      batch_size = 4,
                      num_workers = 2,
                      epochs = 100,
                      learning_rate = 0.001,
                      momentum = 0.99,
                      ):
    print("\nTraining Inception v3")
    import torch.nn.functional as F
    dataset_path = path
    num_classes = len(os.listdir(path+'/train'))

    net = models.inception_v3(pretrained=True)
    net.AuxLogits.fc = nn.Linear(768,num_classes)
    net.fc = nn.Linear(2048,num_classes)
    net = net.cuda()

    import numpy as np
    global mean, std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transform=transforms.Compose([transforms.Resize([299,299]),transforms.ToTensor(), 
                                  transforms.Normalize(mean, std)
                                  ])
    trainset=datasets.ImageFolder(root=path+'/train',transform=transform)
    valset=datasets.ImageFolder(root=path+'/val',transform=transform)

    trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    valloader=torch.utils.data.DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    # Get a batch of training data
    inputs, classes = next(iter(trainloader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    class_names = trainset.classes
    imshow(out, title=[class_names[x] for x in classes])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = learning_rate, momentum = momentum)

    best_val=0
    e=[]
    l=[]
    v=[]

    for epoch in range(epochs): 
        print('Epoch: ', epoch+1) 
        e.append(epoch)
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            #print('data: ', i)
            inputs, labels = data
            inputs,labels=inputs.cuda(),labels.cuda()
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(F.log_softmax(outputs[0]), labels)+criterion(F.log_softmax(outputs[1]), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        #print(running_loss/len(trainloader))
        l.append(running_loss/len(trainloader))
        running_loss = 0.0
        net.eval()
        correct = 0
        total = 0
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            inputs,labels=inputs.cuda(),labels.cuda()

            outputs = net(inputs)
            loss = criterion(F.log_softmax(outputs), labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Validation accuracy: ', 100*correct/total)
        if (epoch == 0):
            best_val=running_loss
            torch.save(net,path+"/inception.h5")  
        elif (running_loss<best_val):
            best_val=running_loss
            torch.save(net,path+"/inception.h5")
        
        v.append(running_loss/len(valloader))
        plot(path, v, l, 'Loss inception_v3')

    # Getting Proba distribution
    print("\nGetting the Probability Distribution")
    testloader=torch.utils.data.DataLoader(valset,batch_size=1)
    net=net.eval()

    correct = 0
    total = 0

    import csv
    import numpy as np
    dst = path+"/csv"
    if not os.path.exists(dst):
        os.makedirs(dst)
    f = open(dst+"/inception_v3.csv",'w+',newline = '')
    writer = csv.writer(f)

    saving = []
    with torch.no_grad():
          num = 0
          temp_array = np.zeros((len(testloader),num_classes))
          for i,data in enumerate(testloader):
              images, labels = data
              sample_fname, _ = testloader.dataset.samples[i]
              labels=labels.cuda()
              outputs = net(images.cuda())
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (predicted == labels.cuda()).sum().item()
              prob = torch.nn.functional.softmax(outputs, dim=1)
              saving.append(sample_fname.split('/')[-1])
              temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
              num+=1
    print("Accuracy = ",100*correct/total)

    for i in range(len(testloader)):
      k = temp_array[i].tolist()
      k.append(saving[i])
      writer.writerow(k)

    f.close()
