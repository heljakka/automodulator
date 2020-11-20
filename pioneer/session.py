import numpy as np
import torch
from torch import nn, optim

import os
import copy

from pioneer.robust_loss_pytorch.adaptive import AdaptiveLossFunction
from pioneer.model import Generator, Discriminator, SpectralNormConv2d, AdaNorm

#TODO:REMOVE:
#from pioneer import config
#args   = config.get_config()

class Session:
    def __init__(self, start_iteration = -1, nz=512, n_label=1, phase=-1, max_phase=7,
                 match_x_metric='robust', lr=0.0001, reset_optimizers=-1, no_progression=False,
                 images_per_stage=2400e3, device=None):
        # Note: 3 requirements for sampling from pre-existing models:
        # 1) Ensure you save and load both Generator and Encoder multi-gpu versions (DataParallel) or both not.
        # 2) Ensure you set the same phase value as the pre-existing model and that your local and global alpha=1.0 are set
        # 3) Sample from the g_running, not from the latest generator

        self.alpha = -1
        self.sample_i = start_iteration
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

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.generator = nn.DataParallel( Generator(self.nz, self.n_label).to(device=self.device) )
        self.g_running = nn.DataParallel( Generator(self.nz, self.n_label).to(device=self.device) )
        self.encoder   = nn.DataParallel( Discriminator(nz = self.nz,
                                                        n_label = self.n_label,
                                                        binary_predictor = False)
                                                        .to(device=self.device) )

        print("Using ", torch.cuda.device_count(), " GPUs!")

        self.adaptive_loss_N = 9

        self.reset_opt()

        print('Session created.')

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
        for layer in SpectralNormConv2d.spectral_norm_layers:
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

    def load(self, path):
        checkpoint = torch.load(path)

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
                        print("LR in optimizer update: {} => {}".format(param_group['lr'], args.lr))
                        param_group['lr'] = self.lr

            print("Reloaded old optimizers")
        else:
            print("Despite loading the state, we reset the optimizers.")

        self.alpha = checkpoint['alpha']

        loaded_phase = int(checkpoint['phase'])
        if self.phase > 0: #If the start phase has been manually set, try to actually use it (e.g. when have trained 64x64 for extra rounds and then turning the model over to 128x128)
            self.phase = min(loaded_phase, self.phase)
            print("Use start phase: {}".format(self.phase))
        if loaded_phase > self.max_phase:
            print('Warning! Loaded model claimed phase {} but max_phase={}'.format(self.phase, args.max_phase))
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

        for layer_i, layer in enumerate(SpectralNormConv2d.spectral_norm_layers):
            setattr(layer, 'weight_u', us_list[layer_i])

    def create(self, save_dir, is_evaluation, force_alpha = -1):
        print(f'sample i: {self.sample_i}')
        if self.sample_i <= 0:
            self.sample_i = 1
            if self.no_progression:
                self.sample_i = int( (self.max_phase + 0.5) * self.images_per_stage ) # Start after the fade-in stage of the last iteration
                force_alpha = 1.0
                print("Progressive growth disabled. Setting start step = {} and alpha = {}".format(self.sample_i, force_alpha))
        else:
            reload_from = '{}/checkpoint/{}_state.pth'.format(save_dir, str(self.sample_i).zfill(6)) #e.g. '604000' #'600000' #latest'   
            print(reload_from)
            if os.path.exists(reload_from):
                requested_checkpoint = self.sample_i
                self.load(reload_from)
                print("Loaded {}".format(reload_from))
                print("Iteration asked {} and got {}".format(requested_checkpoint, self.sample_i))               

                if is_evaluation:
                    self.generator = copy.deepcopy(self.g_running)
            else:
                assert(not is_evaluation)
                print('Start from iteration {}'.format(self.sample_i))

        self.g_running.train(False)

        if force_alpha >= 0.0:
            self.alpha = force_alpha

        if not is_evaluation:
            accumulate(self.g_running, self.generator, 0)

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
        gen_offset = sum(1  for j in Generator.supportBlockPoints if j <= self.phase) #TODO: Replace Generator with self.generator once the g_running is handled properly as well.
        return self.phase - gen_offset

    def getReso(self):
        return 4 * 2 ** self.getResoPhase()

    def getBatchSize(self):
        return batch_size(self.getReso())

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)
