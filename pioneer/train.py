import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torchvision import utils

from pioneer.model import Generator, Discriminator, SpectralNormConv2d, AdaNorm

from datetime import datetime
import random
import copy

import os

from pioneer import config
from pioneer import utils
from pioneer import data
from pioneer import evaluate

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from torch.nn import functional as F

from pioneer.training_scheduler import TrainingScheduler
from pioneer.robust_loss_pytorch import general
from pioneer.robust_loss_pytorch.adaptive import AdaptiveLossFunction

args   = config.get_config()
writer = None

def batch_size(reso):
    if args.gpu_count == 1:
        save_memory = False
        if not save_memory:
            batch_table = {4:128, 8:128, 16:128, 32:64, 64:32, 128:32, 256:16, 512:4, 1024:1}
        else:
            batch_table = {4:64, 8:32, 16:32, 32:32, 64:16, 128:14, 256:2, 512:2, 1024:1}
    elif args.gpu_count == 2:
        batch_table = {4:256, 8:256, 16:256, 32:128, 64:64, 128:28, 256:32, 512:14, 1024:2}
    elif args.gpu_count == 4:
        batch_table = {4:512, 8:256, 16:128, 32:64, 64:32, 128:64, 256:64, 512:32, 1024:4}
    elif args.gpu_count == 8:
        batch_table = {4:512, 8:512, 16:512, 32:256, 64:256, 128:128, 256:64, 512:32, 1024:8}
    else:
        assert(False)
    
    return batch_table[reso]

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

#        import ipdb; ipdb.set_trace()

        self.sample_i = int(checkpoint['iteration'])

        importOldGen = args.import_old_gen_format
        if importOldGen:
            for gen in ['G_state_dict', 'G_running_state_dict']:
                for attr in ['weight', 'bias']:
                    checkpoint[gen]['module.adanorm_blocks.0.mod.{}'.format(attr)] = checkpoint[gen]['module.progression.0.conv.3.mod.{}'.format(attr)]
                    for i in range(0,9):
                        print('module.adanorm_blocks.{}.mod.{}'.format(1+i*2, attr))
                        print("=>")
                        print('module.progression.{}.conv.0.2.mod.{}'.format(i+1,attr))
                        print('module.adanorm_blocks.{}.mod.{}'.format(2+i*2, attr))
                        print("=>")
                        print('module.progression.{}.conv.3.mod.{}'.format(i+1,attr))
                        if 'module.progression.{}.conv.0.2.mod.{}'.format(i+1,attr) in checkpoint[gen]:
                            checkpoint[gen]['module.adanorm_blocks.{}.mod.{}'.format(1+i*2, attr)] = checkpoint[gen]['module.progression.{}.conv.0.2.mod.{}'.format(i+1,attr)]
                            checkpoint[gen]['module.adanorm_blocks.{}.mod.{}'.format(2+i*2, attr)] = checkpoint[gen]['module.progression.{}.conv.3.mod.{}'.format(i+1,attr)]


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

def setup():
    config.init()
    
    utils.make_dirs()
    if not args.testonly:
        config.log_args(args)

    if args.use_TB:
        from dateutil import tz
        from tensorboardX import SummaryWriter
#        from torch.utils.tensorboard import SummaryWriter

        dt = datetime.now(tz.gettz('Europe/Helsinki')).strftime(r"%y%m%d_%H%M")
        global writer
        writer = SummaryWriter("{}/{}/{}".format(args.summary_dir, args.save_dir, dt))

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)   

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

class KLN01Loss(torch.nn.Module): #Adapted from https://github.com/DmitryUlyanov/AGE

    def __init__(self, direction, minimize):
        super(KLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'

        self.direction = direction

    def forward(self, samples):

        assert samples.nelement() == samples.size(1) * samples.size(0), '?'

        samples = samples.view(samples.size(0), -1)

        self.samples_var = utils.var(samples)
        self.samples_mean = samples.mean(0)

        samples_mean = self.samples_mean
        samples_var = self.samples_var

        if self.direction == 'pq':
            t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
            t2 = samples_var.log()

            KL = (t1 + t2 - 0.5).mean()
        else:
            # In the AGE implementation, there is samples_var^2 instead of samples_var^1
            t1 = (samples_var + samples_mean.pow(2)) / 2
            # In the AGE implementation, this did not have the 0.5 scaling factor:
            t2 = -0.5*samples_var.log()

            KL = (t1 + t2 - 0.5).mean()

        if not self.minimize:
            KL *= -1

        return KL

#adaparams = None

def encoder_train(session, real_image, generatedImagePool, batch_N, match_x, stats, kls, margin):
    encoder = session.encoder
    generator = session.generator

    encoder.zero_grad()
    generator.zero_grad()

    x = Variable(real_image).to(device=args.device) #async=(args.gpu_count>1))
    KL_maximizer = KLN01Loss(direction=args.KL, minimize=False)
    KL_minimizer = KLN01Loss(direction=args.KL, minimize=True)

    e_losses = []

    flipInvarianceLayer = args.flip_invariance_layer
    flipX = flipInvarianceLayer > -1 and session.phase >= flipInvarianceLayer
    
    phiAdaCotrain = args.phi_ada_cotrain

    if flipX:
        phiAdaCotrain = True

    #global adaparams
    if phiAdaCotrain:
        for b in generator.module.adanorm_blocks:
            if not b is None and not b.mod is None:
                for param in b.mod.parameters():
                    param.requires_grad_(True)

    if flipX:
        x_in = x[0:int(x.size()[0]/2),:,:,:]
        x_mirror = x_in.clone().detach().requires_grad_(True).to(device=args.device).flip(dims=[3])
        x[int(x.size()[0]/2):,:,:,:] = x_mirror

    real_z = encoder(x, session.getResoPhase(), session.alpha, args.use_ALQ)
    
    if args.use_real_x_KL:
        # KL_real: - \Delta( e(X) , Z ) -> max_e
        if not flipX:
            KL_real = KL_minimizer(real_z) * args.real_x_KL_scale
        else: # Treat the KL div of each direction of the data as separate distributions
            z_in =          real_z[0:int(x.size()[0]/2),:]
            z_in_mirror =   real_z[int(x.size()[0]/2):,:]
            KL_real =  (KL_minimizer(z_in) +
                        KL_minimizer(z_in_mirror)) * args.real_x_KL_scale /2
        e_losses.append(KL_real)

        stats['real_mean'] = KL_minimizer.samples_mean.data.mean().item()
        stats['real_var'] = KL_minimizer.samples_var.data.mean().item()
        stats['KL_real'] = KL_real.data.item()
        kls = "{0:.3f}".format(stats['KL_real'])

    if flipX:
        x_reco_in = utils.gen_seq([ (z_in, 0, flipInvarianceLayer), # Rotation of x_in, the ID of x_in_mirror (= the ID of x_in)
                                    (z_in_mirror, flipInvarianceLayer, -1)],
                                  session.generator, session)

        x_reco_in_mirror = utils.gen_seq([  (z_in_mirror, 0, flipInvarianceLayer),  # Vice versa
                                            (z_in, flipInvarianceLayer, -1)],
                                         session.generator, session)

        if args.match_x_metric == 'robust':
            loss_flip = torch.mean(session.adaptive_loss[session.getResoPhase()].lossfun((x_in - x_reco_in).view(-1, x_in.size()[1]*x_in.size()[2]*x_in.size()[3]) ))  * match_x * 0.2 + \
                        torch.mean(session.adaptive_loss[session.getResoPhase()].lossfun((x_mirror - x_reco_in_mirror).view(-1, x_mirror.size()[1]*x_mirror.size()[2]*x_mirror.size()[3]) ))  * match_x * 0.2
        else:
            loss_flip = (utils.mismatch(x_in, x_reco_in, args.match_x_metric) +
                         utils.mismatch(x_mirror, x_reco_in_mirror, args.match_x_metric))*args.match_x

        loss_flip.backward(retain_graph=True)
        stats['loss_flip'] = loss_flip.data.item()
        stats['x_reconstruction_error'] = loss_flip.data.item()
        print('Flip loss: {}'.format(stats['loss_flip']))

    else:
        if args.use_loss_x_reco:
            recon_x = generator(real_z, None, session.phase, session.alpha)
            # match_x: E_x||g(e(x)) - x|| -> min_e

            if args.match_x_metric == 'robust':
                err_simple = utils.mismatch(recon_x, x, 'L1') * match_x
                err = torch.mean(session.adaptive_loss[session.getResoPhase()].lossfun((recon_x - x).view(-1, x.size()[1]*x.size()[2]*x.size()[3]) ))  * match_x * 0.2
                print("err vs. ROBUST err: {} / {}".format(err_simple, err))
            else:
                err_simple = utils.mismatch(recon_x, x, args.match_x_metric) * match_x
                err = err_simple

            if phiAdaCotrain:
                err.backward(retain_graph=True)
            else:
                e_losses.append(err)
            stats['x_reconstruction_error'] = err.data.item()

    if phiAdaCotrain:
        for b in session.generator.module.adanorm_blocks:
            if not b is None and not b.mod is None:
                for param in b.mod.parameters():
                   param.requires_grad_(False)
   
    if args.use_loss_fake_D_KL:
        # TODO: The following codeblock is essentially the same as the KL_minimizer part on G side. Unify

        mix_ratio = args.stylemix_E #0.25
        mix_N = int(mix_ratio*batch_N)
        z = Variable( torch.FloatTensor(batch_N, args.nz, 1, 1) ).to(device=args.device) #async=(args.gpu_count>1))
        utils.populate_z(z, args.nz+args.n_label, args.noise, batch_N)

        if session.phase > 0 and mix_N > 0:
            alt_mix_z = Variable( torch.FloatTensor(mix_N, args.nz, 1, 1) ).to(device=args.device) #async=(args.gpu_count>1))
            utils.populate_z(alt_mix_z, args.nz+args.n_label, args.noise, mix_N)
            alt_mix_z = torch.cat((alt_mix_z, z[mix_N:,:]), dim=0) if mix_N < z.size()[0] else alt_mix_z
        else:
            alt_mix_z = None

        with torch.no_grad():
            style_layer_begin = np.random.randint(low=1, high=(session.phase+1)) if not alt_mix_z is None else -1
            fake = utils.gen_seq([  (z, 0, style_layer_begin),
                                    (alt_mix_z, style_layer_begin, -1)
            ], generator, session).detach()

        if session.alpha >= 1.0:
            fake = generatedImagePool.query(fake.data)

        fake.requires_grad_()

        # e(g(Z))
        egz = encoder(fake, session.getResoPhase(), session.alpha, args.use_ALQ)

        # KL_fake: \Delta( e(g(Z)) , Z ) -> max_e
        KL_fake = KL_maximizer(egz) * args.fake_D_KL_scale

        if margin > 0.0:
             KL_loss = torch.min(torch.ones_like(KL_fake) * margin/2, torch.max(-torch.ones_like(KL_fake) * margin, KL_real + KL_fake)) # KL_fake is always negative with abs value typically larger than KL_real. Hence, the sum is negative, and must be gapped so that the minimum is the negative of the margin.
        else:
             KL_loss = KL_real + KL_fake
        e_losses.append(KL_loss)

        stats['fake_mean'] = KL_maximizer.samples_mean.data.mean()
        stats['fake_var'] = KL_maximizer.samples_var.data.mean()
        stats['KL_fake'] = -KL_fake.data.item()
        stats['KL_loss'] = KL_loss.data.item()

        kls = "{0}/{1:.3f}".format(kls, stats['KL_fake'])

    # Update e
    if len(e_losses) > 0:
        e_loss = sum(e_losses)                
        e_loss.backward()
        stats['E_loss'] = np.float32(e_loss.data.cpu())

    session.optimizerD.step()

    if flipX:
        session.optimizerA.step() #The AdaNorm params need to be updated separately since they are on "generator side"

    return kls

def KL_of_encoded_G_output(generator, encoder, z, batch_N, session, alt_mix_z, mix_N):
    KL_minimizer = KLN01Loss(direction=args.KL, minimize=True)
    #utils.populate_z(z, args.nz+args.n_label, args.noise, batch_N)

    assert(alt_mix_z is None or alt_mix_z.size()[0] == z.size()[0]) # Batch sizes must match
    mix_style_layer_begin = np.random.randint(low=1, high=(session.phase+1)) if not alt_mix_z is None else -1
    z_intermediates = utils.gen_seq([	(z, 0, mix_style_layer_begin),
                      			(alt_mix_z, mix_style_layer_begin, -1)
    ], generator, session, retain_intermediate_results = True)

    assert(z_intermediates[0].size()[0] == z.size()[0]) # Batch sizes must remain invariant
    fake = z_intermediates[1 if not alt_mix_z is None else 0]
    z_intermediate = z_intermediates[0][:mix_N,:]

    egz = encoder(fake, session.getResoPhase(), session.alpha, args.use_ALQ)

    if mix_style_layer_begin > -1:
        egz_intermediate = utils.gen_seq([(egz[:mix_N,:], 0, mix_style_layer_begin)], generator, session) # Or we could just call generator directly, ofc.
        assert(egz_intermediate.size()[0] == z_intermediate.size()[0])
    else:
        egz_intermediate = z_intermediate = None

    # KL_fake: \Delta( e(g(Z)) , Z ) -> min_g
    return egz, KL_minimizer(egz) * args.fake_G_KL_scale, egz_intermediate, z_intermediate

def decoder_train(session, batch_N, stats, kls, x):
    session.generator.zero_grad()

    if session.phase > 0:
        for p in session.generator.module.to_rgb[session.phase-1].parameters():
            p.requires_grad_(False)
 
    g_losses = []

    mix_ratio = args.stylemix_D
    mix_N = int(mix_ratio*batch_N) if session.phase > 0 else 0
    z = Variable( torch.FloatTensor(batch_N, args.nz, 1, 1) ).to(device=args.device)
    utils.populate_z(z, args.nz+args.n_label, args.noise, batch_N)

    if mix_N > 0:
        alt_mix_z = Variable( torch.FloatTensor(mix_N, args.nz, 1, 1) ).to(device=args.device)
        utils.populate_z(alt_mix_z, args.nz+args.n_label, args.noise, mix_N)
        alt_mix_z = torch.cat((alt_mix_z, z[mix_N:,:]), dim=0) if mix_N < z.size()[0] else alt_mix_z
    else:
        alt_mix_z = None   

    # KL is calculated from the distro of z's that was re-encoded from z and mixed-z
    egz, kl, egz_intermediate, z_intermediate = KL_of_encoded_G_output(session.generator, session.encoder, z, batch_N, session, alt_mix_z, mix_N)

    if args.use_loss_KL_z:
        g_losses.append(kl) # G minimizes this KL
        stats['KL(Phi(G))'] = kl.data.item()
        kls = "{0}/{1:.3f}".format(kls, stats['KL(Phi(G))'])

    # z_diff is calculated only from the regular z (not mixed z)
    if args.use_loss_z_reco: #and (mix_N == 0 or session.phase == 0):
        z_diff = utils.mismatch(egz[mix_N:,:], z[mix_N:,:], args.match_z_metric) * args.match_z # G tries to make the original z and encoded z match #Alternative: [mix_N:,:]
        z_mix_diff = utils.mismatch(egz_intermediate.view([mix_N,-1]), z_intermediate.view([mix_N,-1]), 'L2') if mix_N>0 else torch.zeros(1).cuda()
        if args.intermediate_zreco > 0:
            g_losses.append(z_mix_diff)
        g_losses.append(z_diff)
        stats['z_reconstruction_error'] = z_diff.data.item()
       	if args.intermediate_zreco > 0:
            stats['z_mix_reconstruction_error'] = z_mix_diff.data.item()

    if len(g_losses) > 0:
        loss = sum(g_losses)
        stats['G_loss'] = np.float32(loss.data.cpu())

        loss.backward()

        if False: #For debugging the adanorm blocks
            from model import AdaNorm
            adaparams0 = list(session.generator.adanorm_blocks[0].mod.parameters())
            param_scale  = np.linalg.norm(adaparams0[0].detach().cpu().numpy().ravel())
            print("Ada norm: {} / {}".format(np.linalg.norm(adaparams0[0].grad.detach().cpu().numpy().ravel()), param_scale ))

    session.optimizerG.step()

    return kls


def train(generator, encoder, g_running, train_data_loader, test_data_loader, session, total_steps, train_mode, sched):
    pbar = tqdm(initial=session.sample_i, total = total_steps)
    
    benchmarking = False

    match_x = args.match_x
    generatedImagePool = None

    refresh_dataset   = True
    refresh_imagePool = True
    refresh_adaptiveLoss = False

    # After the Loading stage, we cycle through successive Fade-in and Stabilization stages

    batch_count = 0

    reset_optimizers_on_phase_start = False

    # TODO Unhack this (only affects the episode count statistics anyway):
    if args.data != 'celebaHQ':
        epoch_len = len(train_data_loader(1,4).dataset)
    else:
        epoch_len = train_data_loader._len['data4x4']

    if args.step_offset != 0:
        if args.step_offset == -1:
            args.step_offset = session.sample_i
        print("Step offset is {}".format(args.step_offset))
        session.phase += args.phase_offset
        session.alpha = 0.0

    last_fade_done_at_reso = -1

    while session.sample_i < total_steps:
        #######################  Phase Maintenance ####################### 
        sched.update(session.sample_i)
        sample_i_current_stage = sched.get_iteration_of_current_phase(session.sample_i)

        if sched.phaseChangedOnLastUpdate:
            match_x = args.match_x # Reset to non-matching phase
            refresh_dataset = True
            refresh_imagePool = True # Reset the pool to avoid images of 2 different resolutions in the pool
            refresh_adaptiveLoss = True
            if reset_optimizers_on_phase_start:
                utils.requires_grad(generator)
                utils.requires_grad(encoder)
                generator.zero_grad()
                encoder.zero_grad()
                session.reset_opt()
                print("Optimizers have been reset.")

        reso = session.getReso()

        # If we can switch from fade-training to stable-training
        if sample_i_current_stage >= args.images_per_stage/2:
            if session.alpha < 1.0:
                refresh_dataset = True # refresh dataset generator since no longer have to fade
                last_fade_done_at_reso = reso
                session.alpha = 1
            match_x = args.match_x * args.matching_phase_x
        else:
            match_x = args.match_x

        # We track whether this resolution was already present in the previous stage, which means that it was already faded once.
        if  last_fade_done_at_reso != reso:
            session.alpha = min(1, sample_i_current_stage * 2.0 / args.images_per_stage) # For 100k, it was 0.00002 = 2.0 / args.images_per_stage

        if refresh_adaptiveLoss:
            session.prepareAdaptiveLossForNewPhase()
            refresh_adaptiveLoss = False
        if refresh_dataset:
            train_dataset = data.Utils.sample_data2(train_data_loader, batch_size(reso), reso, session)
            refresh_dataset = False
            print("Refreshed dataset. Alpha={} and iteration={}".format(session.alpha, sample_i_current_stage))
        if refresh_imagePool:
            imagePoolSize = 200 if reso < 256 else 100
            generatedImagePool = utils.ImagePool(imagePoolSize) #Reset the pool to avoid images of 2 different resolutions in the pool
            refresh_imagePool = False
            print('Image pool created with size {} because reso is {}'.format(imagePoolSize, reso))

        ####################### Training init ####################### 
             
        stats = {}
        stats['z_mix_reconstruction_error'] = 0

        try:
            real_image, _ = next(train_dataset)              
        except (OSError, StopIteration):
            train_dataset = data.Utils.sample_data2(train_data_loader, batch_size(reso), reso, session)
            real_image, _ = next(train_dataset)

        ####################### DISCRIMINATOR / ENCODER ###########################

        utils.switch_grad_updates_to_first_of(encoder, generator)
        kls = encoder_train(session, real_image, generatedImagePool, batch_size(reso), match_x, stats, "", margin = sched.m)

        ######################## GENERATOR / DECODER #############################

        if (batch_count + 1) % args.n_critic == 0:
            utils.switch_grad_updates_to_first_of(generator, encoder)
            
            for _ in range(args.n_generator):               
                kls = decoder_train(session, batch_size(reso), stats, kls, real_image.data)

            accumulate(g_running, generator)

        del real_image

        ########################  Statistics ######################## 

        if args.use_TB:
            for key,val in stats.items():                    
                writer.add_scalar(key, val, session.sample_i)
        elif batch_count % 100 == 0:
            print(stats)

        if args.use_TB:
            writer.add_scalar('LOD', session.getResoPhase() + session.alpha, session.sample_i)

        b = session.getBatchSize()
        zr, xr = (stats['z_reconstruction_error'], stats['x_reconstruction_error']) if ('z_reconstruction_error' in stats and 'x_reconstruction_error' in stats) else (0.0, 0.0)
        e = (session.sample_i / float(epoch_len))

        pbar.set_description(
            ('{0}; it: {1}; phase: {2}; b: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; KL(Phi(x)/Phi(G0)/Phi(G1)/Phi(G2)): {7}; z-reco: {8:.2f}; x-reco {9:.3f}; real_var {10:.6f}; fake_var {11:.6f}; z-mix: {12:.4f};').format(batch_count+1, session.sample_i+1, session.phase, b, session.alpha, reso, e, kls, zr, xr, stats['real_var'], stats['fake_var'], float(stats['z_mix_reconstruction_error']))
            )

        pbar.update(batch_size(reso))
        session.sample_i += batch_size(reso) # if not benchmarking else 100
        batch_count += 1

        ########################  Saving ######################## 

        if batch_count % args.checkpoint_cycle == 0 or session.sample_i >= total_steps:
            for postfix in {str(session.sample_i).zfill(6)}: # 'latest'
                session.save_all('{}/{}_state'.format(args.checkpoint_dir, postfix))

            print("Checkpointed to {}".format(session.sample_i))

        ########################  Tests ######################## 

        try:
            evaluate.tests_run(g_running, encoder, test_data_loader, session, writer,
            reconstruction       = (batch_count % 2400 == 0),
            interpolation        = (batch_count % 2400 == 0),
            collated_sampling    = (batch_count % 800 == 0),
            individual_sampling  = False, #(batch_count % (args.images_per_stage/batch_size(reso)/4) == 0),
            metric_eval          = (batch_count % 2500 == 0)
            )
        except (OSError, StopIteration):
            print("Skipped periodic tests due to an exception.")        

    pbar.close()

def makeTS(opts, session):
    print("Using preset training scheduler of celebaHQ and LSUN Bedrooms to set phase length, LR and KL margin. To override this behavior, modify the makeTS() function.")

    ts = TrainingScheduler(opts, session)

    for p in range(0,2):
        ts.add(p*2400, _phase=p, _lr=[0.0005, 0.0005], _margin=0, _aux_operations=None)
#        ts.add(p*2400, _phase=p, _lr=[0.0005, 0.0005], _margin=0.20, _aux_operations=None) # Use this instead if the early stages tend to diverge catastrophically

    for p in range(2,4):
        ts.add(p*2400, _phase=p, _lr=[0.0005, 0.0005], _margin=0.05, _aux_operations=None)

    if args.data == 'celebaHQ':
        ts.add(9600, _phase=4, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
        if args.flip_invariance_layer <=0:
            ts.add(20040, _phase=5, _lr=[0.001, 0.001], _margin=0.02, _aux_operations=None)
            ts.add(27500, _phase=6, _lr=[0.001, 0.001], _margin=0.04, _aux_operations=None)
        else:
            ts.add(12000, _phase=5, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None) #Phase 5 <=> reso 64x64 extension
            ts.add(20040, _phase=6, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None) #Phase 6 <=> reso 128x128
    elif args.data == 'ffhq':
        ts.add(9600, _phase=4, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
        if args.flip_invariance_layer <=0:
            ts.add(20040, _phase=5, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
            ts.add(30680, _phase=6, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
            ts.add(35700, _phase=7, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
        else:
            ts.add(12000, _phase=5, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None) #Phase 5 <=> reso 64x64 extension
            ts.add(20040, _phase=6, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None) #Phase 6 <=> reso 128x128
    elif args.data == 'lsun':
        ts.add(9600, _phase=4, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
        if args.lsunm02e:
            ts.add(20040, _phase=5, _lr=[0.001, 0.001], _margin=0.02, _aux_operations=None)
            ts.add(24840, _phase=6, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
        else:
            ts.add(20040, _phase=5, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
            ts.add(30000, _phase=6, _lr=[0.001, 0.001], _margin=0.05, _aux_operations=None)
            ts.add(38000, _phase=6, _lr=[0.001, 0.001], _margin=0.08, _aux_operations=None)
    elif args.data != 'cifar10':
        for p in range(4,8):
            ts.add(p*2400, _phase=p, _lr=[0.001, 0.001], _margin=0, _aux_operations=None)

    return ts


def main():
    setup()
    session = Session()
    session.create()

    print('PyTorch {}'.format(torch.__version__))

    if args.train_path:
        train_data_loader = data.get_loader(args.data, args.train_path)
    else:
        train_data_loader = None
    
    if args.test_path or args.z_inpath:
        test_data_loader = data.get_loader(args.data, args.test_path)
    elif args.aux_inpath:
        test_data_loader = data.get_loader(args.data, args.aux_inpath)
    else:
        test_data_loader = None

    # 4 modes: Train (with data/train), test (with data/test), aux-test (with custom aux_inpath), dump-training-set
    
    if args.run_mode == config.RUN_TRAIN:
        scheduler = makeTS([session.optimizerD, session.optimizerG], session)

        train(session.generator, session.encoder, session.g_running, train_data_loader, test_data_loader,
            session  = session,
            total_steps = args.total_kimg * 1000,
            train_mode = args.train_mode,
            sched = scheduler)
    elif args.run_mode == config.RUN_TEST:
        #evaluate.Utils.reconstruction_dryrun(session.g_running, session.encoder, test_data_loader, session=session)
        ###evaluate.tests_run(session.generator, session.encoder, test_data_loader, session=session, writer=writer)
        evaluate.tests_run(session.g_running, session.encoder, test_data_loader, session=session, writer=writer)
    elif args.run_mode == config.RUN_DUMP:
        session.phase = args.start_phase
        data.dump_training_set(train_data_loader, args.dump_trainingset_N, args.dump_trainingset_dir, session)

if __name__ == '__main__':
    main()
