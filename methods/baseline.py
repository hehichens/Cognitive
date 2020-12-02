from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.basemodel import CNN_DEAP

class EEGDataset(Dataset):
    
    def __init__(self, x_tensor, y_tensor):
        
        self.x = x_tensor
        self.y = y_tensor
        
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)

class TrainModel():
    def __init__(self, num_classes, data, labels, input_size, learning_rate, batch_size, epoch):
        self.num_classes = num_classes
        self.data = data
        self.labels = labels
        self.input_size = input_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = learning_rate 
        self.batch_size = batch_size
        self.epoch = epoch
    
    def leave_one_subject_out(self):

        subjects = self.data.shape[0]
        
        ACC = []
        for i in range(subjects):
            
            index = np.arange(subjects)
            index_train = np.delete(index, i)
            index_test = i
            
            print(self.data.shape)
            print(self.labels.shape)
            
            # 划分数据集
            data_test = self.data[index_test, :, :, :]
            labels_test = self.labels[index_test, :]
            
            data_train = self.data[index_train, :, :, :]
            labels_train = self.labels[index_train, :]
            
            data_train, labels_train, data_val, labels_val = self.split(data_train, labels_train)
            
            # 增加深度维度
            data_train = data_train[:,np.newaxis,:,:]
            data_val = data_val[:,np.newaxis,:,:]
            data_test = data_test[:,np.newaxis,:,:]
            
            # 转换数据格式
            data_train = torch.from_numpy(data_train).float()
            labels_train = torch.from_numpy(labels_train).long()

            data_val = torch.from_numpy(data_val).float()
            labels_val = torch.from_numpy(labels_val).long()
                
            print(data_test.shape)
            print(labels_test.shape)
                
            data_test = torch.from_numpy(data_test).float()
            labels_test = torch.from_numpy(labels_test).long()
            
            print("Training:", data_train.size(), labels_train.size())
            print("Validation:", data_val.size(), labels_val.size())
            print("Test:", data_test.size(), labels_test.size())
    
            ACC_one_sub = self.train(data_train, labels_train,
                                      data_test, labels_test,
                                      data_val, labels_val)
        
            ACC.append(ACC_one_sub)
        
            print("Subject:" + str(i) +"\nAccuracy:%.2f" % ACC_one_sub)
        
        mean_ACC = np.mean(ACC)
        print("*"*20)
        print("Mean accuracy of model is: %.2f" % mean_ACC)
    
    def split(self, data, label):
        '''将训练数据划分为验证数据和训练数据'''

        np.random.seed(0)
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)

        index = np.arange(data.shape[0])
        index_random = index

        np.random.shuffle(index_random)
        data = data[index_random]
        labels = label[index_random]

        # get validation set
        val_data = data[int(data.shape[0]*0.8):]
        val_labels = labels[int(data.shape[0]*0.8):]

        # get train set
        train_data = data[0:int(data.shape[0]*0.8)]
        train_labels = label[0:int(data.shape[0]*0.8)]

        return train_data, train_labels, val_data, val_labels

    def make_train_step(self, model, loss_fn, optimzier):
        def train_step(x, y):
            model.train()
            yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item()/len(pred)
            loss_r = self.regulization(model, self.Lambda)
            loss = loss_fn(yhat, y)
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()
            return loss.item(), acc
        return train_step
    
    def train(self, train_data, train_label, test_data, test_label, val_data,
              val_label):
        print("Available Device:" + self.device)
        
        model = CNN_DEAP(self.num_classes, self.input_size)
        
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)
        
        train_step = self.make_train_step(model, loss_fn, optimizer)
        
        # 导入数据
        dataset_train = EEGDataset(train_data, train_label)
        dataset_val = EEGDataset(val_data, val_label)
        dataset_test = EEGDataset(test_data, test_label)
        
        # 创建DataLoader
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(dataset=dataset_val, batch_size=self.batch_size,shuffle=True, pin_memory=False)
        test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=True, pin_memory=False)
        
        for epoch in range(self.epoch):
            loss_epoch = []
            acc_epoch = []
            for i, (x_batch, y_batch) in enumerate(train_loader):

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)

            losses.append(sum(loss_epoch)/len(loss_epoch))
            accs.append(sum(acc_epoch)/len(acc_epoch))
            loss_epoch = []
            acc_epoch = []
            print('Epoch [{}/{}], Loss:{:.4f}, Acc:{:.4f}'.format(epoch+1, num_epochs, losses[-1], accs[-1]))

            ##############Validation process####################
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    model.eval()

                    yhat = model(x_val)
                    pred = yhat.max(1)[1]
                    # print("------")
                    # print(pred.size())
                    # print(y_val.size())
                    # print("------")
                    correct = (pred == y_val).sum()
                    acc = correct.item()/len(pred)
                    val_loss = loss_fn(yhat, y_val)
                    val_losses.append(val_loss.item())
                    val_acc.append(acc)

                Acc_val.append(sum(val_acc)/len(val_acc))
                Loss_val.append(sum(val_losses)/len(val_losses))
                print('Evaluation Loss:{:.4f}, Acc:{:.4f}'.format(Loss_val[-1], Acc_val[-1]))
                val_losses = []
                val_acc = []
                
        #########test process############
        model = torch.load('valence_max_model.pt')
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

                model.eval()

                yhat = model(x_test)
                pred = yhat.max(1)[1]
                correct = (pred == y_test).sum()
                acc = correct.item()/len(pred)
                test_loss = loss_fn(yhat, y_test)
                test_losses.append(test_loss.item())
                test_acc.append(acc)

            print('Test Loss:{:.4f}, Acc:{:.4f}'
                  .format(sum(test_losses)/len(test_losses), sum(test_acc)/len(test_acc)))
            Acc_test = (sum(test_acc)/len(test_acc))
        
        return Acc_test


if __name__ == "__main__":
    train = TrainModel(num_classes=2,
                   data=data, 
                   labels=valence_labels,
                   input_size=data[0,0].shape,
                   learning_rate=0.00001,
                   batch_size=50,
                   epoch=5)
    train.leave_one_subject_out()