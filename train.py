import sys
import time 
import os
import torch 
import torch.nn.functional as F 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Transformer
from tokenizer import VOCABULARY
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("runs/model")

def lr_schedule(input_size:int, step: any, warmup_steps: int = 4000) -> float:

    if warmup_steps <= 0:
        step += 4000
        warmup_steps = 4000

    step += 1e-6

    if type(step) == torch.Tensor:
        arg = torch.min(step ** (-0.5), step * (warmup_steps ** (-1.5)))
    else:
        arg = min(step ** (-0.5),step * (warmup_steps ** (-1.5)))

    return (input_size ** (-0.5))*arg 


def print_progress(index: int, total: int, last: str="", fi: str="") -> None:
    spec_char = ["◓","◑","◒","◐"]
    percent = ("{0:.1f}").format(100 * ((index) / total))
    filledLength = 30 * index // total
    bar = '▇' * filledLength + ' ' * (30 - filledLength )
    percent = " "*(5-len(str(percent))) + str(percent)
    if index == total:
        print(f"\r{fi} \033[92m {spec_char[index%4]} |{bar}| {percent}% {last} \033[00m", end="\r\r")
    else:
        print(f"\r{fi} \033[96m {spec_char[index%4]} |{bar}| {percent}% {last} \033[00m", end="\r\r")
    if index == total:
        print()


class Datasets:

    @staticmethod
    def load_data(path: str) -> torch.Tensor:
        data = torch.load(path).long()
        return data

    
    @staticmethod 
    def train_test_split(data: torch.Tensor, test_size: float, batch_size: int) -> list:
        train_len = round(data.shape[0] * (1 - test_size))
        # train_data = data[:train_len]
        train_data = data
        valid_data = data[train_len:]

        train_set = TensorDataset(train_data[:, :-1], train_data[:, 1:])
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        valid_set = TensorDataset(valid_data[:, :-1], valid_data[:, 1:])
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)

        return train_set, train_loader, valid_set, valid_loader


class NeuralNetwork:
 
    def __init__(self: object,
                hyper_params: dict,
                data_path: str,
                batch_size: int,
                check_point_path: str,
                warmup_steps: int = 4000,
                test_size: float = 0.2,
                load_check_point: bool = False,
                device: str = 'cpu') -> None:
        
        self.data_path = data_path 
        self.batch_size = batch_size 
        self.device = device

        data = Datasets.load_data(data_path).to(device)

        if hyper_params["max_position"] > 0:
            hyper_params["max_position"] = max(hyper_params["max_position"], data.shape[-1])
        
        self.train_set, self.train_loader, self.valid_set, self.valid_loader = Datasets.train_test_split(data, test_size, batch_size)

        self.model = Transformer(**hyper_params).to(device)
        self.hyper_params = hyper_params

        self.warmup_steps = warmup_steps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: lr_schedule(hyper_params["d_model"], x, warmup_steps)
        )

        self.check_point_path = check_point_path
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []

        if load_check_point and os.path.isfile(check_point_path):
            self.load_parameters()


    def save_parameters(self: object) -> None:
        parameters = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
            "train_acc": self.train_acc,
            "valid_acc": self.valid_acc,
            "warmup_steps": self.warmup_steps,
            "hyper_params": self.hyper_params
        }
        torch.save(parameters, self.check_point_path)


    def load_parameters(self: object) -> None:
        parameters = torch.load(self.check_point_path)

        del self.model, self.optimizer, self.scheduler

        self.model = Transformer(**parameters["hyper_params"]).to(self.device)
        self.model.load_state_dict(parameters["model_state_dict"])
        self.hyper_params = parameters["hyper_params"]

        self.warmup_steps = parameters["warmup_steps"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.optimizer.load_state_dict(parameters["optimizer_state_dict"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: lr_schedule(self.hyper_params["d_model"], x, self.warmup_steps)
        )
        self.scheduler.load_state_dict(parameters["scheduler_state_dict"])
        self.criterion = F.cross_entropy

        self.train_loss = parameters["train_loss"]
        self.valid_loss = parameters["valid_loss"]
        self.train_acc = parameters["train_acc"]
        self.valid_acc = parameters["valid_acc"]


    def loss_function(self: object,
                      y_predict: torch.Tensor,
                      y: torch.Tensor,
                      criterion: nn.functional) -> float:
        mask = torch.ne(y, torch.zeros_like(y))
        _loss = criterion(y_predict, y, reduction='mean')
        mask = mask.to(_loss.dtype)
        loss = _loss * mask 

        return torch.sum(loss) / torch.sum(mask)


    def accuracy_function(self:object, 
                          y_predict: torch.Tensor,
                          y: torch.Tensor):
        compare_table = (torch.argmax(y_predict, dim=1) == y).float()
        # mean_on_row = torch.mean(compare_table, dim=1)
        accuracy = torch.mean(compare_table, dim=0)

        return accuracy


    def train_batch(self: object, x: torch.Tensor, y: torch.Tensor) -> float:
        y_predict = self.model(x)
        self.optimizer.zero_grad()
        loss = self.loss_function(y_predict.view(y_predict.size(0)*y_predict.size(1), y_predict.size(2)), y.view(y.size(0)*y.size(1)), F.cross_entropy)
        loss.backward()
        acc = self.accuracy_function(y_predict.view(y_predict.size(0)*y_predict.size(1), y_predict.size(2)), y.view(y.size(0)*y.size(1)))
        self.optimizer.step()
        self.scheduler.step()

        return float(loss), float(acc) 


    def valid_batch(self: object, x: torch.Tensor, y: torch.Tensor) -> float:
        y_predict = self.model(x)
        loss = self.loss_function(y_predict.view(y_predict.size(0)*y_predict.size(1), y_predict.size(2)), y.view(y.size(0)*y.size(1)), F.cross_entropy)
        acc = self.accuracy_function(y_predict.view(y_predict.size(0)*y_predict.size(1), y_predict.size(2)), y.view(y.size(0)*y.size(1)))
        return float(loss), float(acc)


    def fit(self: object, epochs: int) -> None:
        try:
            for epoch in range(1, epochs + 1):
                time_st_epoch = time.time()
                
                if epoch%1 == 0:
                    print(f"Epoch: {' '*(len(str(epochs)) - len(str(epoch)))}{epoch}/{epochs}")

                train_epoch_loss = []
                valid_epoch_loss = []
                train_epoch_acc = []
                valid_epoch_acc = []

                self.model.train()
                for id_batch, (X_train, y_train) in enumerate(self.train_loader):
                    st = time.time()
                    X_train = X_train.to(self.device)
                    y_train = y_train.to(self.device)
                    # writer.add_graph(self.model, X_train)
                    loss, acc = self.train_batch(X_train, y_train)
                    train_epoch_loss.append(loss)
                    train_epoch_acc.append(acc)
                    calc_time = round(time.time() - st, 2)
                    if epoch%1 == 0:
                        print_progress(id_batch+1, len(self.train_loader), last=f"{calc_time}s/step - Loss: {round(loss, 5)} - Acc: {round(acc, 2)}", fi="\tTrain process:")

                self.model.eval()
                for id_batch, (X_valid, y_valid) in enumerate(self.valid_loader):
                    st = time.time()
                    X_valid = X_valid.to(self.device)
                    y_valid = y_valid.to(self.device)
                    loss, acc = self.valid_batch(X_valid, y_valid)
                    valid_epoch_loss.append(loss)
                    valid_epoch_acc.append(acc)
                    calc_time = round(time.time() - st, 2)
                    if epoch%1 == 0:
                        print_progress(id_batch+1, len(self.valid_loader), last=f"{calc_time}s/step - Loss: {round(loss, 5)} - Acc: {round(acc, 2)}", fi="\tValid process:")

                train_loss_avg = float(torch.mean(torch.Tensor(train_epoch_loss)))
                valid_loss_avg = float(torch.mean(torch.Tensor(valid_epoch_loss)))
                train_acc_avg = float(torch.mean(torch.Tensor(train_epoch_acc)))
                valid_acc_avg = float(torch.mean(torch.Tensor(valid_epoch_acc)))

                self.train_loss.append(train_loss_avg)
                self.valid_loss.append(valid_loss_avg)
                self.train_acc.append(train_acc_avg)
                self.valid_acc.append(valid_acc_avg)

                if epoch%1 == 0:
                    calc_time = round(time.time() - time_st_epoch, 2)
                    print(f"=> Train loss: {round(train_loss_avg, 5)} - Train acc: {round(train_acc_avg, 2)} - Valid loss: {round(valid_loss_avg, 5)} - Valid acc: {round(valid_acc_avg, 2)}  - Time: {calc_time}s/epoch")
                
                    self.save_parameters()
            self.save_parameters()
        except KeyboardInterrupt:
            self.save_parameters()
            sys.exit()


def trainer(mode: str, epoch: int, batch_size: int, pretrain: bool) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hyper_params = {
        "d_model": 512,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 128,
        "max_relative_dist": 256,
        "max_position": 256,
        "vocab_size": len(VOCABULARY["NOTE"]),
        "dropout": 0.3,
        "norm_eps": 1e-3,
        "bias": True
    }
    data_path = "./data_gen/note_tensor_v3.pt"
    cpk_path = "./checkpoints/model_note_frames_v3.pt"
    if mode == "SYMBOL":
        hyper_params["vocab_size"] = len(VOCABULARY["TIME"])
        data_path = "./data_gen/time_tensor_v3.pt"
        cpk_path = "./checkpoints/model_time_frames_v3.pt"

    model = NeuralNetwork(
        hyper_params=hyper_params,
        data_path=data_path,
        batch_size=batch_size,
        check_point_path=cpk_path,
        warmup_steps=4000,
        test_size=0.3, 
        load_check_point=pretrain,
        device=device
    )
    model.fit(epoch)


if __name__ == "__main__":
    trainer("NOTE",7, 1, False)
    trainer("SYMBOL", 15, 1, False)