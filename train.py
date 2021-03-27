import os
import json
import os
import json
import torch
import wandb
import argparse
from apex import amp
from tqdm import tqdm
from mel2samp import Mel2Samp
from scipy.io.wavfile import write
from torch.autograd import Variable
from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss



class Trainer:
    def __init__(self, logger, directory):
        super(Trainer, self).__init__()
        self.logger = logger
        self.device = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory
        
    def reset(self, train_config: dict,
                    train_data_config: dict,
                    test_data_config: dict,
                    dist_config: dict,
                    waveglow_config: dict):
        
        self.epochs = train_config['epochs']
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        
        torch.manual_seed(train_config['seed'])
        torch.cuda.manual_seed(train_config['seed'])
        self.sigma = train_config['sigma']
        
        self.criterion = WaveGlowLoss(train_config['sigma'])
        self.model = WaveGlow(**waveglow_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config['learning_rate'])
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
        
        trainset = Mel2Samp(**train_data_config)
        testset = Mel2Samp(**test_data_config)
        self.train_loader = DataLoader(trainset, batch_size=train_config['batch_size'], drop_last=True)
        self.test_loader = DataLoader(testset, num_workers=1, batch_size=1, drop_last=False)
        
        if not os.path.isdir(train_config['output_directory']):
            os.makedirs(train_config['output_directory'])
            
        
    
    def run_epoch(self, epoch):
        bar = tqdm(total=len(self.train_loader) * self.train_loader.batch_size , position=0, leave=False, desc=f'{epoch}:loss {0.0}')
        for batch in self.train_loader:
            self.model.zero_grad()

            mel, audio = batch
            outputs = self.model((mel.to(self.device), audio.to(self.device)))

            loss = self.criterion(outputs)

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()
            self.logger.log({'loss: ': loss.item()})
            bar.update(self.train_loader.batch_size)
            bar.set_description(f'{epoch}:loss {loss.item()}')  
            
    def run_test(self, epoch):
        bar = tqdm(total=len(self.test_loader) * self.test_loader.batch_size , position=0, leave=False)
        for i, batch in enumerate(self.test_loader):

            mel, _ = batch
            mel = mel.to(self.device)
            mel = mel.half() 
            
            with torch.no_grad():
                audio = self.model.infer(mel, sigma=0.6)
                audio = audio * 32768.0
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            write(f'{self.directory}/{epoch}_{i}_example.wav', 22050, audio)
            self.logger.log({f'Epoch_{epoch}/{i}_example.wav': [wandb.Audio(audio, caption="Nice", sample_rate=22050)]})
            bar.update(self.test_loader.batch_size)
            
    def save_model(self):
        torch.save(self.model.state_dict(), 'WaveGlow_rus')
        
                        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.run_epoch(epoch)
            self.model.eval()
            self.run_test(epoch)
            if not epoch % 25:
                self.save_model()
        
        

if __name__ == "__main__":
    with open('config.json') as f:
        data = f.read()

    
    config = json.loads(data)
    train_config = config["train_config"]
    train_data_config = config["train_data_config"]
    test_data_config = config["test_data_config"]
    dist_config = config["dist_config"]
    waveglow_config = config["waveglow_config"]
    
    
    wandb.login()
    wandb.init(project="WaveGlow", reinit=True)
   
    trainer = Trainer(wandb, 'examples')
    trainer.reset(train_config, 
                  train_data_config, 
                  test_data_config, 
                  dist_config, 
                  waveglow_config)
    trainer.train()

