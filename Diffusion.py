import torch.nn as nn
import torch
from utils import random_rotation_point_cloud_torch_batch
from Network import *

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999), alphas_cumprod[1:]

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, rnet_config, config, pretrained=False):
        rnet_config.dropout=0.1
        rnet_config.use_grad_checkpoint=True
        super(finetuned_RibonanzaNet, self).__init__(rnet_config)
        if pretrained:
            self.load_state_dict(torch.load(config.pretrained_weight_path,map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)

        decoder_dim=config.decoder_dim
        self.structure_module=[SimpleStructureModule(d_model=decoder_dim, nhead=config.decoder_nhead, 
                 dim_feedforward=decoder_dim*4, pairwise_dimension=rnet_config.pairwise_dimension, dropout=0.0) for i in range(config.decoder_num_layers)]
        self.structure_module=nn.ModuleList(self.structure_module)

        self.xyz_embedder=nn.Linear(3,decoder_dim)
        self.xyz_norm=nn.LayerNorm(decoder_dim)
        self.xyz_predictor=nn.Linear(decoder_dim,3)
        
        self.adaptor=nn.Sequential(nn.Linear(rnet_config.ninp,decoder_dim),nn.LayerNorm(decoder_dim))

        self.distogram_predictor=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
                                                nn.Linear(rnet_config.pairwise_dimension,40))

        self.time_embedder=SinusoidalPosEmb(decoder_dim)

        self.time_mlp=nn.Sequential(nn.Linear(decoder_dim,decoder_dim),
                                    nn.ReLU(),  
                                    nn.Linear(decoder_dim,decoder_dim))
        self.time_norm=nn.LayerNorm(decoder_dim)

        self.distance2pairwise=nn.Linear(1,rnet_config.pairwise_dimension,bias=False)

        self.pair_mlp1=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
                                    nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension*2),
                                    nn.ReLU(),
                                    nn.Linear(rnet_config.pairwise_dimension*2,rnet_config.pairwise_dimension))

        self.pair_mlp2=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
                                    nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension*2),
                                    nn.ReLU(),
                                    nn.Linear(rnet_config.pairwise_dimension*2,rnet_config.pairwise_dimension))

        self.pair_vector_linear=nn.Linear(3,rnet_config.pairwise_dimension,bias=False)

        #hyperparameters for diffusion
        self.n_times = config.n_times

        #self.model = model
        
        # beta_1, beta_T = config.beta_min, config.beta_max
        # betas = torch.linspace(start=beta_1, end=beta_T, steps=config.n_times)#.to(device) # follows DDPM paper
        # self.sqrt_betas = torch.sqrt(betas)
                                     
        # #define alpha for forward diffusion kernel
        # self.alphas = 1 - betas
        # self.sqrt_alphas = torch.sqrt(self.alphas)
        # alpha_bars = torch.cumprod(self.alphas, dim=0)
        # self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
        # self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        # print(self.sqrt_alpha_bars.min(),self.sqrt_alpha_bars.max())

        # plt.plot(self.sqrt_alpha_bars, label='sqrt_alpha_bars')
        # plt.plot(self.sqrt_one_minus_alpha_bars, label='sqrt_one_minus_alpha_bars')
        # plt.plot(self.sqrt_alpha_bars**2+self.sqrt_one_minus_alpha_bars**2, label='sum')
        # plt.legend()
        # plt.savefig('alpha_bars.png')
        # plt.clf()
        # exit()


        #cosine schedule
        self.betas, self.alpha_bars = cosine_beta_schedule(config.n_times)
        self.betas, self.alpha_bars = torch.tensor(self.betas).float(), torch.tensor(self.alpha_bars).float()

        self.alpha_bars = self.alpha_bars.clip(0.001, 0.999)

        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-self.alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)

        # plt.plot(self.sqrt_alpha_bars, label='sqrt_alpha_bars')
        # plt.plot(self.sqrt_one_minus_alpha_bars, label='sqrt_one_minus_alpha_bars')
        # plt.plot(self.sqrt_alpha_bars**2+self.sqrt_one_minus_alpha_bars**2, label='sum')
        # plt.legend()
        # plt.savefig('cosine_alpha_bars.png')
        # print(self.sqrt_alpha_bars.min(),self.sqrt_alpha_bars.max())
        # exit()
        self.data_std=config.data_std


    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward
    
    def embed_pair_distance(self,inputs):
        pairwise_features,xyz=inputs
        vector_matrix=(xyz[:,None,:,:]-xyz[:,:,None,:])#*self.data_std

        distance_matrix=(vector_matrix**2).sum(-1)
        distance_matrix=1/(1+distance_matrix)
        distance_matrix=distance_matrix[:,:,:,None]
        pairwise_features=pairwise_features+\
                          self.distance2pairwise(distance_matrix)+\
                          self.pair_vector_linear(vector_matrix)

        pairwise_features+=self.pair_mlp1(pairwise_features)
        pairwise_features+=self.pair_mlp2(pairwise_features)

        return pairwise_features

    def forward(self,src,xyz,t):
        
        #with torch.no_grad():
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        
        distogram=self.distogram_predictor(pairwise_features)

        sequence_features=self.adaptor(sequence_features)

        decoder_batch_size=xyz.shape[0]
        sequence_features=sequence_features.repeat(decoder_batch_size,1,1)
        

        pairwise_features=pairwise_features.expand(decoder_batch_size,-1,-1,-1)

        pairwise_features= checkpoint.checkpoint(self.custom(self.embed_pair_distance), [pairwise_features,xyz],use_reentrant=False)

        time_embed=self.time_embedder(t).unsqueeze(1)
        tgt=self.xyz_norm(sequence_features+self.xyz_embedder(xyz)+time_embed)

        tgt=self.time_norm(tgt+self.time_mlp(tgt))

        for layer in self.structure_module:
            #tgt=layer([tgt, sequence_features,pairwise_features,xyz,None])
            tgt=checkpoint.checkpoint(self.custom(layer),
            [tgt, sequence_features,pairwise_features,xyz,None],
            use_reentrant=False)
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        
        xyz=self.xyz_predictor(tgt).squeeze(0)
        #.squeeze(0)

        return xyz, distogram
    

    def denoise(self,sequence_features,pairwise_features,xyz,t):
        decoder_batch_size=xyz.shape[0]
        sequence_features=sequence_features.expand(decoder_batch_size,-1,-1)
        pairwise_features=pairwise_features.expand(decoder_batch_size,-1,-1,-1)

        pairwise_features=self.embed_pair_distance([pairwise_features,xyz])

        sequence_features=self.adaptor(sequence_features)
        time_embed=self.time_embedder(t).unsqueeze(1)
        tgt=self.xyz_norm(sequence_features+self.xyz_embedder(xyz)+time_embed)
        tgt=self.time_norm(tgt+self.time_mlp(tgt))
        #xyz_batch_size=xyz.shape[0]
        


        for layer in self.structure_module:
            tgt=layer([tgt, sequence_features,pairwise_features,xyz,None])
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        xyz=self.xyz_predictor(tgt).squeeze(0)
        # print(xyz.shape)
        # exit()
        return xyz


    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1
    
    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5
    
    def make_noisy(self, x_zeros, t): 
        # assume we get raw data, so center and scale by 35
        x_zeros = x_zeros - torch.nanmean(x_zeros,1,keepdim=True)
        x_zeros = x_zeros/self.data_std
        #rotate randomly
        x_zeros = random_rotation_point_cloud_torch_batch(x_zeros)


        # perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)
        epsilon = torch.randn_like(x_zeros).to(x_zeros.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars.to(x_zeros.device), t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars.to(x_zeros.device), t, x_zeros.shape)
        
        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
    
        return noisy_sample.detach(), epsilon
    
    
    # def forward(self, x_zeros):
    #     x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        
    #     B, _, _, _ = x_zeros.shape
        
    #     # (1) randomly choose diffusion time-step
    #     t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(x_zeros.device)
        
    #     # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
    #     perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        
    #     # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
    #     pred_epsilon = self.model(perturbed_images, t)
        
    #     return perturbed_images, epsilon, pred_epsilon
    
    
    def denoise_at_t(self, x_t, sequence_features, pairwise_features, timestep, t):
        B, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(sequence_features.device)
        else:
            z = torch.zeros_like(x_t).to(sequence_features.device)
        
        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred = self.denoise(sequence_features, pairwise_features, x_t, timestep)
        
        alpha = self.extract(self.alphas.to(x_t.device), timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas.to(x_t.device), timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars.to(x_t.device), timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas.to(x_t.device), timestep, x_t.shape)
        
        # denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
        
        return x_t_minus_1#.clamp(-1., 1)
                
    def sample(self, src, N):
        # start from random noise vector, NxLx3
        x_t = torch.randn((N, src.shape[1], 3)).to(src.device)
        
        # autoregressively denoise from x_T to x_0
        #     i.e., generate image from noise, x_T

        #first get conditioning
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        # sequence_features=sequence_features.expand(N,-1,-1)
        # pairwise_features=pairwise_features.expand(N,-1,-1,-1)
        distogram=self.distogram_predictor(pairwise_features).squeeze()
        distogram=distogram.squeeze()[:,:,2:40].softmax(-1)*torch.arange(2,40).float().cuda() 
        distogram=distogram.sum(-1)  

        for t in range(self.n_times-1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(src.device)
            x_t = self.denoise_at_t(x_t, sequence_features, pairwise_features, timestep, t)
        
        # denormalize x_0 into 0 ~ 1 ranged values.
        #x_0 = self.reverse_scale_to_zero_to_one(x_t)
        x_0 = x_t * self.data_std
        return x_0, distogram

    def sample_heun(self, src, N, num_steps=None):
        """
        Heun's method sampler with optional fewer steps (coarse stepping).
        
        Args:
            src (torch.Tensor): Input sequence tensor.
            N (int): Number of samples to generate.
            num_steps (int, optional): Number of timesteps to sample with. If None, uses full DDPM schedule.
        """
        device = src.device
        x_t = torch.randn((N, src.shape[1], 3)).to(device)

        # Get conditioning
        sequence_features, pairwise_features = self.get_embeddings(src, torch.ones_like(src).long().to(device))
        distogram = self.distogram_predictor(pairwise_features).squeeze()
        distogram = distogram.squeeze()[:, :, 2:40] * torch.arange(2, 40).float().to(device)
        distogram = distogram.sum(-1)

        if num_steps is None:
            timesteps = list(range(self.n_times - 1, 0, -1))
        else:
            timesteps = torch.linspace(self.n_times - 1, 1, steps=num_steps, dtype=torch.long).tolist()
        # print(timesteps)
        # exit()
        for i in range(len(timesteps)):
            t = int(timesteps[i])
            t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0

            t_curr = torch.full((N,), t, dtype=torch.long).to(device)
            t_next_tensor = torch.full((N,), t_next, dtype=torch.long).to(device)

            alpha_bar_t = self.extract(self.sqrt_alpha_bars.to(device), t_curr, x_t.shape) ** 2
            alpha_bar_next = self.extract(self.sqrt_alpha_bars.to(device), t_next_tensor, x_t.shape) ** 2

            # Predict noise at x_t
            eps1 = self.denoise(sequence_features, pairwise_features, x_t, t_curr)

            # # Predict x_0
            # x_0 = (x_t - eps1 * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)

            # # Euler step
            # x_t_euler = torch.sqrt(alpha_bar_next) * x_0 + torch.sqrt(1 - alpha_bar_next) * eps1
            # x_t = x_t_euler
            # # # Predict noise at x_{t-1}
            # eps2 = self.denoise(sequence_features, pairwise_features, x_t_euler, t_next_tensor)

            # # # Heun step (second-order correction)
            # eps_avg = 0.5 * (eps1 + eps2)
            # #x_0 = (x_t - eps_avg * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)
            # x_t = torch.sqrt(alpha_bar_next) * x_0 + torch.sqrt(1 - alpha_bar_next) * eps_avg

            # x_t = (x_t - eps1 * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t) * torch.sqrt(alpha_bar_next) + \
            #     torch.sqrt(1 - alpha_bar_next) * eps1

            

            # x_t = x_t/ torch.sqrt(alpha_bar_next) * torch.sqrt(alpha_bar_next) - \
            #     eps1 * (torch.sqrt((1 - alpha_bar_next)*alpha_bar_next) / torch.sqrt(alpha_bar_t) + torch.sqrt(1 - alpha_bar_next))

            # Euler step
            scale_factor = torch.sqrt(alpha_bar_next)/torch.sqrt(alpha_bar_next)
            step_size = (torch.sqrt((1 - alpha_bar_next)*alpha_bar_next) / torch.sqrt(alpha_bar_t) + torch.sqrt(1 - alpha_bar_next))

            x_t_euler = x_t * scale_factor - eps1 * step_size

            x_t = x_t_euler
            # x_t_euler = batched_svd_align(x_t_euler, x_t)[0]

            # # Compute the noise prediction for the next step
            # if i + 1 < len(timesteps):

            #     eps2 = self.denoise(sequence_features, pairwise_features, x_t_euler, t_next_tensor)
            #     # Heun step (second-order correction)
            #     eps_avg =  (0.75 * eps1 + 0.25 * eps2)
            #     x_t = x_t * scale_factor - eps_avg * step_size
            # else:
            #     x_t = x_t_euler

            # x_t = x_t/ torch.sqrt(alpha_bar_next) * torch.sqrt(alpha_bar_next) - \
            #     eps1 * (torch.sqrt((1 - alpha_bar_next)*alpha_bar_next) / torch.sqrt(alpha_bar_t) + torch.sqrt(1 - alpha_bar_next))



        x_0 = x_t * self.data_std
        return x_0, distogram