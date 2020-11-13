import numpy as np
import torch
from torch import nn, optim

import os
import copy

from pioneer.robust_loss_pytorch.adaptive import AdaptiveLossFunction
from pioneer.model import Generator, Discriminator, SpectralNormConv2d, AdaNorm

#TODO:REMOVE:
from pioneer import config
args   = config.get_config()

class Session:
    def __init__(self):
        # Note: 4 requirements for sampling from pre-existing models:
        # 1) Ensure you save and load both multi-gpu versions (DataParallel) or both not.
        # 2) Ensure you set the same phase value as the pre-existing model and that your local and global alpha=1.0 are set
        # 3) Sample from the g_running, not from the latest generator
        # 4) You may need to warm up the g_running by running evaluate.reconstruction_dryrun() first

        self.alpha = -1
        self.sample_i = min(args.start_iteration, 0)
        self.phase = args.start_phase

        self.generator = nn.DataParallel( Generator(args.nz+1, args.n_label).to(device=args.device) )
        self.g_running = nn.DataParallel( Generator(args.nz+1, args.n_label).to(device=args.device) )
        self.encoder   = nn.DataParallel( Discriminator(nz = args.nz+1, n_label = args.n_label, binary_predictor = False).to(device=args.device) )

        print("Using ", torch.cuda.device_count(), " GPUs!")

        self.adaptive_loss_N = 9

        self.reset_opt()

        print('Session created.')

    def reset_opt(self):
        self.adaptive_loss = []
        adaptive_loss_params = []
        if args.match_x_metric == 'robust':
            for j in range(self.adaptive_loss_N): #Assume 9 phases: 4,8,16,32,64,128,256, 512, 1024 ... ²
                loss_j = (AdaptiveLossFunction(num_dims = 3*2**(4+2*j), float_dtype=np.float32, 
                #device='cuda:0'))
                device=args.device))
                self.adaptive_loss.append(loss_j)
                adaptive_loss_params += list(loss_j.parameters())

        self.optimizerG = optim.Adam(self.generator.parameters(), args.lr, betas=(0.0, 0.99))
        self.optimizerD = optim.Adam(list(self.encoder.parameters()) + adaptive_loss_params, args.lr, betas=(0.0, 0.99)) # includes all the encoder parameters...
        
        _adaparams = np.array([list(b.mod.parameters()) for b in self.generator.module.adanorm_blocks]).flatten() #list(AdaNorm.adanorm_blocks[0].mod.parameters())

        self.optimizerA = optim.Adam(_adaparams, args.lr, betas=(0.0, 0.99))

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

        if args.reset_optimizers <= 0:
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
                    if param_group['lr'] != args.lr:
                        print("LR in optimizer update: {} => {}".format(param_group['lr'], args.lr))
                        param_group['lr'] = args.lr

            print("Reloaded old optimizers")
        else:
            print("Despite loading the state, we reset the optimizers.")

        self.alpha = checkpoint['alpha']
        self.phase = int(checkpoint['phase'])
        if args.start_phase > 0: #If the start phase has been manually set, try to actually use it (e.g. when have trained 64x64 for extra rounds and then turning the model over to 128x128)
            self.phase = min(args.start_phase, self.phase)
            print("Use start phase: {}".format(self.phase))
        if self.phase > args.max_phase:
            print('Warning! Loaded model claimed phase {} but max_phase={}'.format(self.phase, args.max_phase))
            self.phase = args.max_phase
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

    def create(self):
        if args.start_iteration <= 0:
            args.start_iteration = 1
            if args.no_progression:
                self.sample_i = args.start_iteration = int( (args.max_phase + 0.5) * args.images_per_stage ) # Start after the fade-in stage of the last iteration
                args.force_alpha = 1.0
                print("Progressive growth disabled. Setting start step = {} and alpha = {}".format(args.start_iteration, args.force_alpha))
        else:
            reload_from = '{}/checkpoint/{}_state.pth'.format(args.save_dir, str(args.start_iteration).zfill(6)) #e.g. '604000' #'600000' #latest'   
            print(reload_from)
            if os.path.exists(reload_from):
                self.load(reload_from)
                print("Loaded {}".format(reload_from))
                print("Iteration asked {} and got {}".format(args.start_iteration, self.sample_i))               

                if args.testonly:
                    self.generator = copy.deepcopy(self.g_running)
            else:
                assert(not args.testonly)
                self.sample_i = args.start_iteration
                print('Start from iteration {}'.format(self.sample_i))

        self.g_running.train(False)

        if args.force_alpha >= 0.0:
            self.alpha = args.force_alpha

        if not args.testonly:
            accumulate(self.g_running, self.generator, 0)

    def prepareAdaptiveLossForNewPhase(self):
        if args.match_x_metric != 'robust':
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
