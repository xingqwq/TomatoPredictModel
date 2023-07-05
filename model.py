import torch
import torch.nn as nn
          
class TomatoModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device, drop_out = 0.5, layer_num = 1, isBidirectional = False):
        super(TomatoModel, self).__init__()
        self.input_size = input_size + 16
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.device = device
        self.isBidirectional = isBidirectional
        self.drop_out = drop_out
        
        # CNN提取图片特征
        self.pic_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=11, stride=4, padding=0), #16*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), #16*13*13

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), #32*13*13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),#32*6*6
        )
        self.pic_linear = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(32*6*6, 16),
            nn.ReLU(),
        )
        # 定义内部参数
        # self.all_weight = []
        # for _ in range(self.layer_num):
        self.input_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.forget_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.c_gate = nn.Linear(self.input_size  + self.hidden_size, self.hidden_size)
        self.h_gate = nn.Linear(self.input_size  + self.hidden_size, self.hidden_size)
        self.output_gate = nn.Linear(self.hidden_size+self.input_size, self.hidden_size)
        if self.isBidirectional == True:
            self.classifier = nn.Linear(self.hidden_size*2, self.output_size)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def __str__(self) -> str:
        return "TomatoModel"

    def forward(self, datas, img, dataLens):
        _x = None
        # 提取图片特征
        for i in range(img.shape[0]):
            _img = self.pic_features(img[i])
            _img = torch.flatten(_img, start_dim=1)
            _img = self.pic_linear(_img)
            if _x == None:
                _x = torch.hstack((_img, datas[i])).unsqueeze(0)
            else:
                _x = torch.vstack((_x, torch.hstack((_img, datas[i])).unsqueeze(0)))
        x = nn.utils.rnn.pack_padded_sequence(_x, dataLens, batch_first=True, enforce_sorted=False)
        # 时序预测
        if self.isBidirectional == False:
            hidden = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            c = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            output = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            index = 0
            i = 0
            
            while i < x.data.shape[0]: 
                input = x.data[i:i+x.batch_sizes[index]]
                id = x.sorted_indices[:x.batch_sizes[index]]
                combined = torch.cat((input,hidden[id]),dim=1)
                f_t = self.sigmoid(self.forget_gate(combined))
                i_t = self.sigmoid(self.input_gate(combined))
                c_hat = self.tanh(self.c_gate(combined))
                c[id] = f_t*c[id]+i_t*c_hat
                output[id] = self.sigmoid(self.output_gate(combined))
                hidden[id] = output[id]*self.tanh(c[id])
                i += x.batch_sizes[index]
                index += 1
            y = self.classifier(hidden)
        
            return y, hidden

        else:
            # 正向
            hidden = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            c = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            output = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            index = 0
            i = 0
            
            while i < x.data.shape[0]: 
                input = x.data[i:i+x.batch_sizes[index]]
                id = x.sorted_indices[:x.batch_sizes[index]]
                combined = torch.cat((input,hidden[id]),dim=1)
                f_t = self.sigmoid(self.forget_gate(combined))
                i_t = self.sigmoid(self.input_gate(combined))
                c_hat = self.tanh(self.c_gate(combined))
                c[id] = f_t*c[id]+i_t*c_hat
                output[id] = self.sigmoid(self.output_gate(combined))
                hidden[id] = output[id]*self.tanh(c[id])
                i += x.batch_sizes[index]
                index += 1
                
            # 反向
            hidden_ = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            c_ = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            output_ = torch.zeros((x.sorted_indices.shape[0], self.hidden_size)).to(self.device)
            index = len(x.batch_sizes)-1
            i = len(x.data)
            
            while i >0: 
                input = x.data[i-x.batch_sizes[index]:i]
                id = x.sorted_indices[:x.batch_sizes[index]]
                combined = torch.cat((input,hidden_[id]),dim=1)
                f_t = self.sigmoid(self.forget_gate(combined))
                i_t = self.sigmoid(self.input_gate(combined))
                c_hat = self.tanh(self.c_gate(combined))
                c_[id] = f_t*c_[id]+i_t*c_hat
                output_[id] = self.sigmoid(self.output_gate(combined))
                hidden_[id] = output_[id]*self.tanh(c_[id])
                i -= x.batch_sizes[index]
                index -= 1
            o = torch.cat((hidden,hidden_),dim=1)
            y = self.classifier(o)
        
            return y, o