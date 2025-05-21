# Code reference(Colab CIFAR10 Diffusion): https://colab.research.google.com/drive/1IJkrrV-D7boSCLVKhi7t5docRYqORtm3
from contextlib import contextmanager
from copy import deepcopy
import math
import torch
from torch import optim, nn


# Utilities
@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the model (a residual U-Net)
class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.ReLU(inplace=True),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class Diffusion(nn.Module):
    def __init__(self, name):
        super().__init__()
        in_dim = 64  # The feature dimension

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        
        if name == "CIFAR10":
            self.class_embed = nn.Embedding(10, 4)
            self.class_emb_num = 4
        elif name == "CIFAR100":
            self.class_embed = nn.Embedding(100, 20)
            self.class_emb_num = 20
        else:
            raise ValueError("Invalid name: {}".format(name))  # 为了防止输入的name既不是"CIFAR10"，也不是"CIFAR100"

        self.net = nn.Sequential(
            nn.Linear(64 + self.class_emb_num + 16, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
        )
    # v = model(noised_reals, log_snrs, classes)
    def forward(self, input, log_snrs, cond):
        timestep_embed, class_embed = self.timestep_embed(log_snrs[:, None]), self.class_embed(cond)
        concat = torch.cat([input, class_embed, timestep_embed], dim=1)
        return self.net(concat)


# Define the noise schedule and sampling loop
def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given the log SNR for a timestep."""
    return log_snrs.sigmoid().sqrt(), log_snrs.neg().sigmoid().sqrt()


def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -torch.special.expm1(1e-4 + 10 * t**2).log()


@torch.no_grad()
def sample(model, x, steps, classes, eta = 0.):
    """Draws samples from a model given starting noise."""
    # ts: timestep embedding
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in range(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * log_snrs[i], classes).float()

        # Predict the noise and the denoised image
        pred_img = x * alphas[i] - v * sigmas[i]   # denoised image
        eps = x * sigmas[i] + v * alphas[i]    # predict noise

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred_img * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred_img

'''
def eval_loss(model, rng, reals, classes, device):
    # Draw uniformly distributed continuous timesteps
    t = rng.draw(reals.shape[0])[:, 0].to(device)

    # Calculate the noise schedule parameters for those timesteps
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)
    weights = log_snrs.exp() / log_snrs.exp().add(1)

    # Combine the ground truth images and the noise
    alphas = alphas[:, None]
    sigmas = sigmas[:, None]
    noise = torch.randn_like(reals)
    noised_reals = reals * alphas + noise * sigmas
    targets = noise * alphas - reals * sigmas

    # Compute the model output and the loss.   ------ non weighted
    with torch.cuda.amp.autocast():
        v = model(noised_reals, log_snrs, classes)
        return (v - targets).pow(2).mean([1]).mul(weights).mean()
'''






### 修改处
def eval_loss(
    model,                # The diffusion model (predicting v)
    rng,                  # Random number generator for timesteps (e.g., karras_diffusion.sampling.GaussianSampler)
    reals,                # Batch of clean input features (z0). Shape: [batch, feature_dim]
    classes,              # Class labels for the batch. Shape: [batch]
    class_means,          # Tensor of pre-calculated class means. Shape: [num_classes, feature_dim]
    global_mean,          # Tensor of the pre-calculated global mean. Shape: [1, feature_dim] or [feature_dim]
    beta,                 # Weight for the Intra-NC loss (L_Intra)
    gamma,                # Weight for the Inter-NC loss (L_Inter2)
    device                # Torch device
    ):
    """
    Calculates the diffusion model loss including Neural Collapse regularization,
    based on the formulas from loss.pptx.

    Args:
        model: The diffusion model (parameterized for v-prediction).
        rng: Random number generator for timesteps.
        reals: Batch of clean input features (z0).
        classes: Class labels for the batch.
        class_means: Tensor of pre-calculated class means for all classes.
        global_mean: Tensor of the pre-calculated global mean feature.
        beta: Weight for the Intra-NC loss.
        gamma: Weight for the Inter-NC loss.
        device: Torch device.

    Returns:
        Scalar tensor representing the total combined loss.
    """
    beta = 1e-4
    gamma = 1e-4
    # --- Step 1: Standard Diffusion Setup ---

    # Draw uniformly distributed continuous timesteps
    # Note: karras sampler uses log-normal, but ppt formulas average t~[1,T]
    # Using uniform continuous time for flexibility as in the original snippet
    t = rng.draw(reals.shape[0])[:, 0].to(device) # Shape: [batch]

    # Calculate the noise schedule parameters (alphas, sigmas) based on time t
    # This depends on the specific schedule used (e.g., VP, VE, EDM's logSNR)
    # Assuming functions that return alpha_t and sigma_t for the v-pred formula
    # z_t = alpha_t * z0 + sigma_t * noise
    # v_target = alpha_t * noise - sigma_t * z0
    log_snrs = get_ddpm_schedule(t)  # Example: Get logSNR based on time
    alphas, sigmas = get_alphas_sigmas(log_snrs) # Get alpha_t, sigma_t
    # Ensure shapes are [batch, 1, ...] for broadcasting

    # Calculate timestep-dependent weights (as in the original snippet)
    # w(t) = SNR(t) / (SNR(t) + 1)
    weights = log_snrs.exp() / log_snrs.exp().add(1)

    # Sample noise and create noisy input z_t
    noise = torch.randn_like(reals)                 # epsilon ~ N(0, I)
    noised_reals = reals * alphas + noise * sigmas  # z_t

    # Calculate the target for v-prediction
    v_targets = noise * alphas - reals * sigmas     # v_target

    # --- Step 2: Model Prediction and Base Loss ---

    # Compute the model output (predicted v)
    # Model takes noisy input, time embedding (log_snrs), and class condition
    with torch.cuda.amp.autocast(): # Use mixed precision if available
        v_pred = model(noised_reals, log_snrs, classes) # Predicted v

        # Calculate the base v-prediction loss (equivalent to L_LDM)
        # MSE between predicted v and target v
        # Average over feature dimensions, keep batch dimension
        ## loss_v_samples = (v_pred - v_targets).pow(2).mean(dim=list(range(1, reals.dim()))) # Shape: [batch]

        # Apply timestep weighting to the base loss samples
        ## weighted_loss_v = loss_v_samples * weights
        ## base_loss = weighted_loss_v.mean() # Average over batch
        base_loss = (v_pred - v_targets).pow(2).mean([1]).mul(weights).mean()
        # --- Step 3: Calculate Predicted Clean Features (z0_hat) ---
        # Required for NC losses. Derived from z_t and predicted v:
        # Formula: z0_hat = alpha_t * z_t - sigma_t * v_pred
        predicted_reals = noised_reals * alphas - v_pred * sigmas # Shape: [batch, feature_dim]

        # --- Step 4: Neural Collapse Loss Calculation ---

        # L_Intra: Intra-class Collapse Loss (Predicted z0 vs Class Mean)
        # Get class means corresponding to batch labels
        # Ensure class_means is on the correct device
        batch_class_means = class_means.to(device)[classes] # Shape: [batch, feature_dim]
        # Calculate squared distance to class mean for each sample
        loss_intra_samples = (predicted_reals - batch_class_means).pow(2).mean(dim=list(range(1, reals.dim()))) # Shape: [batch]
        # Apply the same timestep weighting? Seems reasonable as z0_hat quality depends on t
        weighted_loss_intra = loss_intra_samples * weights
        loss_intra = weighted_loss_intra.mean() # Average over batch

        # L_Inter2: Inter-class Equal Norm Loss (Variance of centered z0_hat norms)
        # Center predicted features using the global mean
        # Ensure global_mean is on the correct device and broadcastable
        centered_predicted_reals = predicted_reals - global_mean.to(device)
        # Calculate L2 norms for each sample's centered prediction
        norms = centered_predicted_reals.norm(p=2, dim=1) # Shape: [batch]
        # Calculate the variance of these norms within the batch.
        # This approximates the goal of making norms equal across classes.
        # We don't apply timestep weight here as variance is a batch statistic.
        loss_inter2 = norms.var()

        # --- Step 5: Total Loss ---
        # Combine base diffusion loss with weighted NC losses
        total_loss = base_loss + beta * loss_intra + gamma * loss_inter2

    return total_loss

# --- Placeholder Helper Functions (Replace with actual implementations) ---
# These depend heavily on the specific diffusion schedule chosen (e.g., VP, VE, EDM)
'''
def get_ddpm_schedule(t):
    """Placeholder: Calculates logSNR based on continuous time t."""
    # Example: Linear beta schedule -> alpha_bar calculation -> logSNR
    # This needs to match the schedule used by the model/paper
    # For simplicity, returning dummy values related to t.
    # A common schedule for continuous time is EDM's log-normal sampling for sigma,
    # then logsnr = -2 * sigma.log(), but the loss uses uniform t.
    # Let's map uniform t in [0,1] to a logSNR range, e.g., [-10, 10]
    min_log_snr = -10.0
    max_log_snr = 10.0
    return max_log_snr - t * (max_log_snr - min_log_snr)


def get_alphas_sigmas(log_snrs):
    """Placeholder: Converts logSNR to alpha_t, sigma_t for z_t = a*z0 + s*noise."""
    # alpha_t^2 = sigmoid(log_snr) , sigma_t^2 = sigmoid(-log_snr)
    # alpha_t = sqrt(sigmoid(log_snr)), sigma_t = sqrt(sigmoid(-log_snr))
    alphas_sq = torch.sigmoid(log_snrs)
    sigmas_sq = torch.sigmoid(-log_snrs)
    return alphas_sq.sqrt(), sigmas_sq.sqrt()
'''
# --- Example Usage ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Assuming model, rng, class_means, global_mean are properly initialized
# B = 16 # Batch size
# F = 128 # Feature dimension
# N = 10 # Num classes
# model = YourDiffusionModel().to(device) # Replace with actual model
# rng = YourTimeSampler() # Replace with actual sampler
# reals = torch.randn(B, F, device=device)
# classes = torch.randint(0, N, (B,), device=device)
# class_means = torch.randn(N, F, device=device) # Load pre-computed means
# global_mean = class_means.mean(dim=0, keepdim=True) # Example global mean
# beta_nc = 1e-4
# gamma_nc = 1e-4

# loss = eval_loss_with_nc(model, rng, reals, classes, class_means, global_mean, beta_nc, gamma_nc, device)
# print(f"Total Loss: {loss.item()}")
# loss.backward() # Example backward pass







def diffusion_train_colab(dataloader_train, dataloader_test, feature_dataloader_tr, dataset_info, dset_info, args):
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(123)
    ema_decay = 0.998
    diffusion_model = Diffusion(args.dataset).to(device)
    diffusion_model_ema = deepcopy(diffusion_model)
    print('Model parameters:', sum(p.numel() for p in diffusion_model.parameters()))
    rng = torch.quasirandom.SobolEngine(1, scramble=True)

    optimizer_diffusion = optim.Adam(diffusion_model.parameters(), lr=0.0001)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(200):
        diffusion_model_ema.train()
        diffusion_loss_sum = 0.0

        for batch, data in enumerate(feature_dataloader_tr):
            # unpack imgs and features
            feature, feature_label = data

            # Transfer data to target device
            feature = feature.to(device)
            feature_label = feature_label.to(device)

            optimizer_diffusion.zero_grad()

            # Evaluate the diffusion loss
            diffusion_loss = eval_loss(diffusion_model, rng, feature, feature_label, device)

            # Backpropagation
            # Do the optimizer step and EMA update
            scaler.scale(diffusion_loss).backward()
            scaler.step(optimizer_diffusion)
            diffusion_loss_sum += diffusion_loss.item()
            ema_update(diffusion_model, diffusion_model_ema, 0.95 if epoch < 20 else ema_decay)
            scaler.update()

        print("epoch: " + str(epoch) + "\n" + "diffusion loss is: ", diffusion_loss_sum / (batch+1))

    # generate features
    diffusion_model_ema.eval()
    sample_batch = 10000
    if args.dataset == "CIFAR10":
        num_per_class = int(sample_batch / 10)
        noise = torch.randn([sample_batch, 64], device=device)
        fakes_classes = torch.arange(10, device=device).repeat_interleave(num_per_class, 0)
    elif args.dataset == "CIFAR100":
        num_per_class = int(sample_batch / 100)
        noise = torch.randn([sample_batch, 64], device=device)
        fakes_classes = torch.arange(100, device=device).repeat_interleave(num_per_class, 0)

    steps = 1000
    samples = sample(diffusion_model_ema, noise, steps, fakes_classes)

    return samples, fakes_classes


