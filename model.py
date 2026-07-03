from lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from utils import EER



# UTILS

def orthogonality_regularization(weights, lambda_reg=1e-6):
    if weights is None:
        return 0
    real_weights = weights.real
    imag_weights = weights.imag

    flattened_real_weights = real_weights.view(real_weights.size(0), real_weights.size(1), -1)
    flattened_imag_weights = imag_weights.view(imag_weights.size(0), imag_weights.size(1), -1)

    gram_matrix_real = torch.matmul(flattened_real_weights, flattened_real_weights.transpose(1, 2))
    gram_matrix_imag = torch.matmul(flattened_imag_weights, flattened_imag_weights.transpose(1, 2))

    identity = torch.eye(flattened_real_weights.size(1), device=weights.device)

    ortho_loss_real = torch.sum((gram_matrix_real - identity)**2)
    ortho_loss_imag = torch.sum((gram_matrix_imag - identity)**2)

    ortho_loss = ortho_loss_real + ortho_loss_imag

    return lambda_reg * ortho_loss

def l1_regularization(weights, lambda_reg=1e-6):
    if weights is None:
        return 0
    real_weights = weights.real
    imag_weights = weights.imag
    l1_loss = torch.sum(torch.abs(real_weights)) + torch.sum(torch.abs(imag_weights))
    return lambda_reg * l1_loss




class AdaptiveComplexBeamformer(nn.Module):
    def __init__(self, num_mics, num_freq_bins, hidden_dim):
        super(AdaptiveComplexBeamformer, self).__init__()
        self.num_mics = num_mics
        self.num_freq_bins = num_freq_bins

        # Parameter Estimator Network (processes multi-channel STFT directly)
        # Input channels = num_mics, output channels = real + imaginary weights
        self.param_estimator = nn.Sequential(
            nn.Conv2d(num_mics * 2, hidden_dim, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(),
            nn.Conv2d(hidden_dim, 2*num_mics, kernel_size=3, padding='same')  # Output real+imag weights
        )
        
    def forward(self, stft_audio):
        stft_audio = stft_audio.permute(1,0,2,3)  # [batch, num_mics, freq_bins, time]
        stft_real_imag = torch.view_as_real(stft_audio)  # Shape: [batch, num_mics, freq_bins, time, 2]
        stft_real_imag = stft_real_imag.permute(0, 1, 4, 2, 3)  # [batch, num_mics, 2, freq_bins, time]
        stft_real_imag = stft_real_imag.flatten(1, 2)  # Merge real/imag as separate input channels: [batch, num_mics*2, freq_bins, time]
        # Pass multi-channel input through the parameter estimator
        weight_map = self.param_estimator(stft_real_imag)  # [batch, 2 * num_mics, time, freq_bins]
        
        # Separate real and imaginary parts
        real_weights, imag_weights = torch.chunk(weight_map, 2, dim=1)  # Each: [batch, num_mics, freq_bins, time]
        complex_weights = torch.complex(real_weights, imag_weights)

        weighted_sum = torch.einsum('bmft,bmft->bft', stft_audio, complex_weights)
        return weighted_sum, complex_weights
    


class AudioDeepFakeDetectionModel(nn.Module):
    def __init__(self, input_channels, n_fft, hop_length, fs=44100):
        super(AudioDeepFakeDetectionModel, self).__init__()
        self.input_channels = input_channels
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.bm_weights = AdaptiveComplexBeamformer(num_mics=self.input_channels, num_freq_bins=self.n_fft//2 + 1, hidden_dim=64)
        self.tf_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)
        
        
        self.heatmap = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 16,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(in_channels = 16, out_channels = 64, 
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            )  # from TASLP-WASPA paper for environmental cues
        

        # CNN model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 1), padding="same")
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), padding="same")
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), padding="same")
        self.bn3 = nn.BatchNorm2d(128)

        self.max_pool1 = nn.MaxPool2d(kernel_size=(8, 1))
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(8, 1))

        self.max_pool2 = nn.MaxPool2d(kernel_size=(8, 1))
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(8, 1))

        self.max_pool3 = nn.MaxPool2d(kernel_size=(4, 1))
        self.avg_pool3 = nn.AvgPool2d(kernel_size=(4, 1))

        # RNN
        self.gru1 = nn.GRU(input_size=(self.n_fft//2)//2, hidden_size=64, bidirectional=True, batch_first = True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, bidirectional=True, batch_first = True)

        # Linear mapping

        self.fc = nn.Linear(in_features=128, out_features=2)
        self.activation = nn.ELU()



    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        stft_audio = [self.tf_transform(mic) for mic in x]
        stft_audio = torch.stack(stft_audio, dim=1) # Shape: [num_mics, batch, freq_bins, time]
        weighted_sum, complex_weights = self.bm_weights(stft_audio)
        _mag2 = weighted_sum.real ** 2 + weighted_sum.imag ** 2  # |z|² — grad is 2·re, 2·im (safe)
        _mag  = torch.sqrt(_mag2 + 1e-12)                        # |z| with safe sqrt
        features = torch.stack([
            torch.log(_mag2 + 1e-6),                 # log-power  (no abs in gradient path)
            weighted_sum.imag / (_mag + 1e-8),        # sin(angle) = imag / |z|
            weighted_sum.real / (_mag + 1e-8),        # cos(angle) = real / |z|
        ], dim=1)

        hm = self.heatmap(features)
        features = features * hm


        # CNN
        features = self.activation(self.bn1(self.conv1(features)))
        features = self.max_pool1(features) + self.avg_pool1(features)

        features = self.activation(self.bn2(self.conv2(features)))  
        features = self.max_pool2(features) + self.avg_pool2(features)

        features = self.activation(self.bn3(self.conv3(features)))
        features = self.max_pool3(features) + self.avg_pool3(features)

        features = features.permute(0, 3, 2, 1)
        features = features.reshape(features.shape[0], features.shape[1], -1)

        features, _ = self.gru1(features)
        features, _ = self.gru2(features)

        latent = features[:, -1, :]
        logits = self.fc(latent)

        return logits, complex_weights, latent

class AudioDeepFakeDetectionModelModule(LightningModule):
    def __init__(self, input_channels, lr, train_label_1, train_label_0, n_fft, hop_length, fs=44100, info=None):
        super().__init__()
        self.save_hyperparameters()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fs = fs
        self.model = AudioDeepFakeDetectionModel(input_channels, self.n_fft, self.hop_length, self.fs)
        self.lr = lr
        self.info = info
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([float(train_label_1), float(train_label_0)]))
        self.eer_computation_train = EER()
        self.eer_computation_val = EER()
        self.eer_computation_test = EER()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['waveform'], batch['label']
        outputs, complex_weights, _ = self(inputs)
        loss = self.criterion(outputs, labels) + orthogonality_regularization(complex_weights) + l1_regularization(complex_weights)
        self.eer_computation_train.update(outputs.detach(), labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('train_eer', self.eer_computation_train, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['waveform'], batch['label']
        outputs, complex_weights, _ = self(inputs)
        loss = self.criterion(outputs, labels) + orthogonality_regularization(complex_weights) + l1_regularization(complex_weights)
        self.eer_computation_val.update(outputs.detach(), labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_eer', self.eer_computation_val, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch['waveform'], batch['label']
        outputs, complex_weights, _ = self(inputs)
        loss = self.criterion(outputs, labels) + orthogonality_regularization(complex_weights) + l1_regularization(complex_weights)
        self.eer_computation_test.update(outputs.detach(), labels)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_err', self.eer_computation_test, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_ortho_loss', orthogonality_regularization(complex_weights), prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_l1_loss', l1_regularization(complex_weights), prog_bar=True, on_epoch=True, on_step=False)
        return loss


    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return {
           "optimizer": opt,
           "lr_scheduler": {
               "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=100, eta_min=0.1*float(self.lr))
                           },
              }
    






if __name__ == '__main__':
    # Definition of a two-channel audio deepfake detection model
    model = AudioDeepFakeDetectionModel(input_channels=2, n_fft=2048, hop_length=1024, fs=44100) # <----- 46 ms of window length with 50 % overlap at 44.1kHz
    # example of a 1 second audio sample with two channels
    sample = torch.randn(1, 2, 44100)
    # example of forward pass
    output, complex_weights, latent = model(sample)
    # output of the model (two neurons, index 0 is the logit for genuine, index 1 is the logit for attack)
    print(output.shape)
    # output of the model (beamforming weights used to compute single-channel audio)
    print(complex_weights.shape)
    # output of latent space
    print(latent.shape)
    # just to visualize the number of paramenters + memory usage
    summary(model, input_size = (1, 2, 44100))

    
    