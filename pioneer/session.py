import numpy as np
import torch
from torch import nn, optim

import os
import copy

from pioneer.robust_loss_pytorch.adaptive import AdaptiveLossFunction
#from pioneer.model import Generator, Discriminator, SpectralNormConv2d, AdaNorm
import pioneer.model
from torchvision import transforms

from torch.autograd import Variable
from torch import randn
from pioneer.utils import normalize

def batch_size(reso):
    gpus = torch.cuda.device_count()
    if gpus == 1:
        save_memory = False
        if not save_memory:
            batch_table = {4:128, 8:128, 16:128, 32:64, 64:32, 128:32, 256:16, 512:4, 1024:1}
        else:
            batch_table = {4:64, 8:32, 16:32, 32:32, 64:16, 128:14, 256:2, 512:2, 1024:1}
    elif gpus == 2:
        batch_table = {4:256, 8:256, 16:256, 32:128, 64:64, 128:28, 256:32, 512:14, 1024:2}
    elif gpus == 4:
        batch_table = {4:512, 8:256, 16:128, 32:64, 64:32, 128:64, 256:64, 512:32, 1024:4}
    elif gpus == 8:
        batch_table = {4:512, 8:512, 16:512, 32:256, 64:256, 128:128, 256:64, 512:32, 1024:8}
    else:
        assert(False)
    
    return batch_table[reso]

class Session:
    def __init__(self, pretrained=False, start_iteration = -1, nz=512, n_label=1, phase=-1, max_phase=7,
                 match_x_metric='robust', lr=0.0001, reset_optimizers=-1, no_progression=False,
                 images_per_stage=2400e3, device=None, force_alpha=-1, save_dir=None, transform_key=None,
                 arch=None):
        # Note: 3 requirements for sampling from pre-existing models:
        # 1) Ensure you save and load both Generator and Encoder multi-gpu versions (DataParallel) or both not.
        # 2) Ensure you set the same phase value as the pre-existing model and that your local and global alpha=1.0 are set
        # 3) Sample from the g_running, not from the latest generator

        self.alpha = -1
        self.requested_start_iteration = start_iteration
        self.sample_i = max(1, start_iteration)
        self.nz = nz
        self.n_label = n_label
        self.phase = phase
        self.max_phase = max_phase
        self.images_per_stage = images_per_stage

        self.match_x_metric = match_x_metric
        self.lr = lr
        self.reset_optimizers = reset_optimizers
        self.no_progression = no_progression

        self.device = device
        self.transform_key = transform_key

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.generator = nn.DataParallel( pioneer.model.Generator(self.nz, self.n_label, arch=arch).to(device=self.device) )
        self.g_running = nn.DataParallel( pioneer.model.Generator(self.nz, self.n_label, arch=arch).to(device=self.device) )
        self.encoder   = nn.DataParallel( pioneer.model.Discriminator(nz = self.nz,
                                                        n_label = self.n_label,
                                                        binary_predictor = False)
                                                        .to(device=self.device) )

        print("Using ", torch.cuda.device_count(), " GPUs!")

        self.adaptive_loss_N = 9

        self.reset_opt()

        if self.requested_start_iteration <= 0 and self.no_progression:
            self.sample_i = int( (self.max_phase + 0.5) * self.images_per_stage ) # Start after the fade-in stage of the last iteration
            self.alpha = 1.0
            print("Progressive growth disabled. Setting start step = {} and alpha = {}".format(self.sample_i, force_alpha))

        # You can construct this model first and the load the model in later, or if pretrained=True, we load it here already:
        if pretrained:
            self.create(save_dir=save_dir,force_alpha=force_alpha)

        print('Session created.')

    def tf(self):
        maxReso = 512 if self.transform_key == 'ffhq512' else 256
        return transforms.Compose([
                        transforms.Resize(maxReso),
                        transforms.CenterCrop(maxReso),
                        transforms.Resize(self.getReso()),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    def reset_opt(self):
        self.adaptive_loss = []
        adaptive_loss_params = []
        if self.match_x_metric == 'robust':
            for j in range(self.adaptive_loss_N): #Assume 9 phases: 4,8,16,32,64,128,256, 512, 1024 ... ²
                loss_j = (AdaptiveLossFunction(num_dims = 3*2**(4+2*j), float_dtype=np.float32, 
                #device='cuda:0'))
                device=self.device))
                self.adaptive_loss.append(loss_j)
                adaptive_loss_params += list(loss_j.parameters())

        self.optimizerG = optim.Adam(self.generator.parameters(), self.lr, betas=(0.0, 0.99))
        self.optimizerD = optim.Adam(list(self.encoder.parameters()) + adaptive_loss_params, self.lr, betas=(0.0, 0.99)) # includes all the encoder parameters...
        
        _adaparams = np.array([list(b.mod.parameters()) for b in self.generator.module.adanorm_blocks]).flatten() #list(AdaNorm.adanorm_blocks[0].mod.parameters())

        self.optimizerA = optim.Adam(_adaparams, self.lr, betas=(0.0, 0.99))

    def save_all(self, path):
        # Spectral Norm layers used to be stored separately for historical reasons.
        us = []
        for layer in pioneer.model.SpectralNormConv2d.spectral_norm_layers:
            us += [getattr(layer, 'weight_u')]

        save_dict = {'G_state_dict': self.generator.state_dict(),
                    'D_state_dict': self.encoder.state_dict(),
                    'G_running_state_dict': self.g_running.state_dict(),
                    'optimizerD': self.optimizerD.state_dict(),
                    'optimizerG': self.optimizerG.state_dict(),
                    'optimizerA': self.optimizerA.state_dict(),
                    'iteration': self.sample_i,
                    'phase': self.phase,
                    'alpha': self.alpha,
                    'SNU': us}
        for i,loss_param in enumerate(self.adaptive_loss):
            save_dict['adaptive_loss_{}'.format(i)] = loss_param.state_dict(),
        torch.save(save_dict, path)
        print("Adaptive Losses saved.")

    def _load(self, checkpoint):
        self.sample_i = int(checkpoint['iteration'])

        loadGeneratorWithNoise = True
        if loadGeneratorWithNoise:
            self.generator.module.create()
            self.g_running.module.create()
            print("Generator dynamic components loaded via create().")

        self.generator.load_state_dict(checkpoint['G_state_dict'])
        self.g_running.load_state_dict(checkpoint['G_running_state_dict'])
        self.encoder.load_state_dict(checkpoint['D_state_dict'])

        if not loadGeneratorWithNoise:
            self.generator.module.create()
            self.g_running.module.create()
        print("Generator dynamic components loaded via create().")

        if self.reset_optimizers <= 0:
            self.optimizerD.load_state_dict(checkpoint['optimizerD'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])
            opts = [self.optimizerD, self.optimizerG]
            try:
                self.optimizerA.load_state_dict(checkpoint['optimizerA'])
                opts += [self.optimizerA] 
            except:
                print('Optimizer for AdaNorm not loaded from state.')
            for opt in opts:
                for param_group in opt.param_groups:
                    if param_group['lr'] != self.lr:
                        print("LR in optimizer update: {} => {}".format(param_group['lr'], self.lr))
                        param_group['lr'] = self.lr

            print("Reloaded old optimizers")
        else:
            print("Despite loading the state, we reset the optimizers.")

        self.alpha = checkpoint['alpha']

        loaded_phase = int(checkpoint['phase'])
        if self.phase > 0: #If the start phase has been manually set, try to actually use it (e.g. when have trained 64x64 for extra rounds and then turning the model over to 128x128)
            self.phase = min(loaded_phase, self.phase)
            print("Use start phase: {}".format(self.phase))
        else:
            self.phase = loaded_phase
        if loaded_phase > self.max_phase:
            print('Warning! Loaded model claimed phase {} but max_phase={}'.format(self.phase, self.max_phase))
            self.phase = self.max_phase
        akeys_ok = True

        for i in range(self.adaptive_loss_N):
            akey = 'adaptive_loss_{}'.format(i)
            if akey in checkpoint:
                self.adaptive_loss[i].load_state_dict(checkpoint[akey][0])
            else:
                akeys_ok = False

        if akeys_ok:
            print("Adaptive Losses loaded.")
        else:
            print('WARNING! Adaptive Loss parameters were not found in checkpoint. Loss calculations will resume without historical information.')

        print("Load SNU from the model file")
        us_list = checkpoint['SNU']
        print(f'Found {len(us_list)} SNU entries')

        for layer_i, layer in enumerate(pioneer.model.SpectralNormConv2d.spectral_norm_layers):
            setattr(layer, 'weight_u', us_list[layer_i])


    # If evaluation mode, then pretrained=True by assumption and generator <- g_running.
    # In not evaluation, then vice versa.

    def eval(self, useLN = True):
        self.g_running.module.use_layer_noise = useLN
        self.generator = copy.deepcopy(self.g_running)
        pioneer.model.SpectralNorm.eval()

    def train(self):
        accumulate(self.g_running, self.generator, 0)
        pioneer.model.SpectralNorm.train()

    # Wraps the load operations
    def create(self, save_dir, force_alpha = -1):
        is_remote_load = save_dir.find('http') != -1

        if not is_remote_load:
            if self.requested_start_iteration > 1:
                reload_from = '{}/checkpoint/{}_state.pth'.format(save_dir, str(self.requested_start_iteration).zfill(6)) #e.g. '604000' #'600000' #latest'   
                print(reload_from)
                if os.path.exists(reload_from):
                    self._load(torch.load(reload_from))
                    print("Loaded {}".format(reload_from))
                    print("Iteration asked {} and got {}".format(self.requested_start_iteration, self.sample_i))               
                else:
                    print('Start from iteration {} without pre-loading!'.format(self.sample_i))
        else:
            print(f'Remote load from {save_dir}')
            self._load(torch.hub.load_state_dict_from_url(save_dir, progress=True))

        self.g_running.train(False)

        if force_alpha >= 0.0:
            self.alpha = force_alpha

    def prepareAdaptiveLossForNewPhase(self):
        if self.match_x_metric != 'robust':
            return
        assert(self.phase>0)
        with torch.no_grad():
            if self.alpha <= 0.02: #Only run the preparatino if this phase has not already been training. In some resum-from-checkpoint scenarios, there might otherwise be misplaced initialization.
                for offset in range(4): #Since the resolution doubles for the next phase, there are 4 new params to stand for each old param. The arrays are flattened, so every 4 slots on the new array correspond to 1 in the old. We copy them over.
                    self.adaptive_loss[self.getResoPhase()].latent_scale[0][offset::4] = self.adaptive_loss[self.getResoPhase()-1].latent_scale[0]
                    self.adaptive_loss[self.getResoPhase()].latent_alpha[0][offset::4] = self.adaptive_loss[self.getResoPhase()-1].latent_alpha[0]
                print('Adaptive loss values have been copied over from phase {} to phase {}'.format(self.phase-1, self.phase))

    def getResoPhase(self):
        gen_offset = sum(1  for j in pioneer.model.Generator.supportBlockPoints if j <= self.phase) #TODO: Replace Generator with self.generator once the g_running is handled properly as well.
        return self.phase - gen_offset

    def getReso(self):
        return 4 * 2 ** self.getResoPhase()

    def getBatchSize(self):
        return batch_size(self.getReso())

    # Evaluation Helpers

    def reconstruct(session, imgs):
        with torch.no_grad():
            return self.decode( self.encode(imgs) )

    def generate(self, num_imgs):
        myz = Variable(torch.randn(num_imgs, 512)).cuda()
        myz = pioneer.utils.normalize(myz)
        
        return self.g_running(
            myz,
            None,
            self.phase,
            1.0).detach()

    def encode(self, imgs):
        if imgs.shape.__len__() == 3:
            imgs = imgs.unsqueeze(0)
        return self.encoder(imgs, self.getResoPhase(), 1.0)

    def decode(self, z):
        if isinstance(z, ScaledBuilder):
            return pioneer.utils.gen_seq(z.z_seq, self.g_running, self).detach()
        else:
            if z.shape.__len__() == 1:
                z = z.unsqueeze(0)
            return self.g_running(z, None, self.getResoPhase(), 1.0).detach()

    def zbuilder(self, **kwargs):
        """ We allow accessing the ScaledBuilder here so that it can be accessed via the loaded Session via PyTorch Hub
        """

        return ScaledBuilder(**kwargs)



def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

class ScaledBuilder():
    _supported_max_stack_size = 9
    def __init__(self, batch_size=1, nz = 512):
        self._z_stack = Variable(torch.randn(batch_size, 1, nz).repeat(1,  ScaledBuilder._supported_max_stack_size, 1)).cuda()
            #torch.randn(batch_size, ScaledBuilder._supported_max_stack_size, nz)).cuda()
        for i in range( ScaledBuilder._supported_max_stack_size ):
            self._z_stack[:,i,:] = normalize(self._z_stack[:,i,:]) #Normalize each modulator separately

    def use(self, z, mod_range):
        assert(len(mod_range) == 2 and mod_range[0] >= -1 and mod_range[1] <= ScaledBuilder._supported_max_stack_size)
        if mod_range[1] == -1:
            mod_range[1] = ScaledBuilder._supported_max_stack_size

        with torch.no_grad():
            self._z_stack[:,range(*mod_range),:] = z

        return self

    def hi(self, z):
        with torch.no_grad():
            self._z_stack[:,:2,:] = z
        return self
    def mid(self, z):
        with torch.no_grad():
            self._z_stack[:,2:4,:] = z
        return self
    def lo(self, z):
        with torch.no_grad():
            self._z_stack[:,5:,:] = z
        return self

    @property
    def z(self):
        return self._z_stack

    @property
    def z_seq(self):
        return [(self._z_stack[:,i,:], i, i+1) for i in range(ScaledBuilder._supported_max_stack_size)] #into the format expected by utils.gen_seq
