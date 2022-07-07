import torch
import torch.nn as nn
import torch.nn.functional as F

features = 64
# define a simple linear VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=14161, out_features=8192)
        self.enc2 = nn.Linear(in_features=8192, out_features=4096)
        self.enc3 = nn.Linear(in_features=4096, out_features=2048)
        self.enc4 = nn.Linear(in_features=2048, out_features=1024)
        self.enc5 = nn.Linear(in_features=1024, out_features=512)
        self.enc6 = nn.Linear(in_features=512, out_features=256)
        self.enc7 = nn.Linear(in_features=256, out_features=180)
        self.enc8 = nn.Linear(in_features=180, out_features=features*2)

        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=180)
        self.dec2 = nn.Linear(in_features=180, out_features=256)
        self.dec3 = nn.Linear(in_features=256, out_features=512)
        self.dec4 = nn.Linear(in_features=512, out_features=1024)
        self.dec5 = nn.Linear(in_features=1024, out_features=2048)
        self.dec6 = nn.Linear(in_features=2048, out_features=4096)
        self.dec7 = nn.Linear(in_features=4096, out_features=8192)
        self.dec8 = nn.Linear(in_features=8192, out_features=14161)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))
        x = F.relu(self.enc7(x))
        
        x = self.enc8(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))
        x = F.relu(self.dec7(x))
        
        reconstruction = torch.sigmoid(self.dec8(x))
        return reconstruction, mu, log_var
