import torch

from torch import nn
from torch.nn import init, DataParallel
from torch.nn import functional as F
from torch.autograd import Variable

from math import sqrt

from pioneer import utils
from pioneer import config

import numpy as np

args   = config.get_config()

def init_linear(linear, lr_gain):
    if lr_gain == 1.0: # Affine mappers
        init.xavier_normal(linear.weight, 1.0)
        linear.bias.data.fill_(1)
    else: # FC-Mapping layers
        init.xavier_normal(linear.weight_orig, 1.0 / lr_gain)
        linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SpectralNorm:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def eval():
        SpectralNorm._is_eval = True

    @staticmethod
    def train():
        SpectralNorm._is_eval = False

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')

        u = getattr(module, self.name + '_u')
        
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.to(device=torch.device('cuda'))
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        # When a state SNU is reloaded (always the case in test mode), we do not want to change weight_u matrix anymore. This caused a bug in style decoder, though the same did not occur for the classical architecture. The symptom: Only the very first decoding run works properly, any batch after that will have its weights collapse to some static value (essentially always the same static image gets generated).
        if not SpectralNorm._is_eval:
            setattr(module, self.name + '_u', u)

def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        gain = getattr(module, self.name + '_gain')

        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in) * gain.data

    @staticmethod
    def apply(module, name, gain):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_parameter(name + '_gain', nn.Parameter(torch.ones(1)*gain))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight', gain=1.0):
    EqualLR.apply(module, name, gain)

    return module

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)

class AdaNorm(nn.Module):
    disable = False
    
    def __init__(self, nz, outz, holder):
        super().__init__()

        if outz == -1:
            self.outz = nz
        else:
            self.outz = outz

        self.mod = nn.Linear(nz, outz*2)
        init_linear(self.mod, lr_gain=1.0)

        holder.adanorm_blocks.append((self))

    def update(self, input_mod):
        self.A = self.mod.to(input_mod.get_device())(input_mod).unsqueeze(2).unsqueeze(2) # The affine transform depends on the actual latent vector


    def forward(self, out):
        global debug_ada
        if debug_ada:
            import ipdb
            ipdb.set_trace()

        o1 = out - out.flatten(start_dim=2).mean(dim=2).unsqueeze(2).unsqueeze(2)
        o2 = o1 / out.flatten(start_dim=2).std(dim=2).unsqueeze(2).unsqueeze(2)
        out = self.A[:,:(self.outz),:,:] * o2 + self.A[:,(self.outz):,:,:]

        if AdaNorm.disable: #For debugging, see the constant output with this.
            out = o2

        return out


class SpectralNormConv2d(nn.Module):
    spectral_norm_layers = nn.ModuleList()
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        init.kaiming_normal(conv.weight)
        conv.bias.data.zero_()
        self.conv = spectral_norm(conv)

        SpectralNormConv2d.spectral_norm_layers.append(self.conv)

    def forward(self, input):
        return self.conv(input)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

def blur2d(x): #Adapted from StyleGAN TF version
    f=[1,2,1]
    f = np.array(f, dtype=np.float32)

    normalize=True
    flip=True
    stride=1
    strides = [1, 1, stride, stride]

    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]    
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[np.newaxis, np.newaxis, :, :]
    f = np.tile(f, [int(x.shape[1]), 1, 1, 1])

    orig_dtype = x.dtype

    nin = np.shape(x)[1]

    return F.conv2d(x, torch.from_numpy(f).cuda(), groups=nin, padding=1)

class BlurLayer(nn.Module):
    def __init__(self):
        super(BlurLayer, self).__init__()

    def forward(self, input):       
        return blur2d(input)

    def backward(self, output):
        return blur2d(output)

class NoiseLayer(nn.Module):
    def __init__(self, mysize):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.zeros(mysize, requires_grad=True))

    def forward(self, input):
        return input + torch.normal(torch.zeros_like(input), torch.ones_like(input)) * self.noise_scale.view((1,-1,1,1))

class NopLayer(nn.Module):
    def __init__(self):
        super(NopLayer, self).__init__()
    def forward(self, input):
        return input

debug_i = 0

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 padding,
                 kernel_size2=None, padding2=None,
                 pixel_norm=False, spectral_norm=False, ada_norm=True, const_layer=False, holder=None, last_act = nn.LeakyReLU(0.2)):
        super().__init__()

        if last_act is None:
            last_act = nn.LeakyReLU(1.0)  #NopLayer()

        #BLUR HACK
        blur = (args.upsampling != 'bilinear')

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.const_layer = const_layer

        if spectral_norm:
            if not args.blurSN:
                self.conv = nn.Sequential(SpectralNormConv2d(in_channel,
                                                             out_channel, kernel1,
                                                             padding=pad1),
                                          nn.LeakyReLU(0.2),
                                          SpectralNormConv2d(out_channel,
                                                             out_channel, kernel2,
                                                             padding=pad2),
                                          last_act)
            else:
                self.conv = nn.Sequential(SpectralNormConv2d(in_channel,
                                                             out_channel, kernel1,
                                                             padding=pad1),
                                          BlurLayer(),
                                          nn.LeakyReLU(0.2),
                                          SpectralNormConv2d(out_channel,
                                                             out_channel, kernel2,
                                                             padding=pad2),
                                          last_act)

        else:
            if ada_norm: # In PGGAN, activation came after PixelNorm. In StyleGAN, PixelNorm/AdaNorm comes after activation.
                ada_conv2D = EqualConv2d # nn.Conv2d

                print("AdaNorm layer count: {}".format(len(holder.adanorm_blocks)))

                maybeBlur = BlurLayer() if blur else nn.Sequential()

                if not const_layer:
                    if not blur:
                        firstBlock = nn.Sequential(ada_conv2D(in_channel, out_channel,
                                                          kernel1, padding=pad1),
                                              nn.LeakyReLU(0.2),
                                              AdaNorm(args.nz+args.n_label, out_channel, holder=holder))
                    else:
                        firstBlock = nn.Sequential(ada_conv2D(in_channel, out_channel,
                                                          kernel1, padding=pad1),
                                              BlurLayer(),
                                              nn.LeakyReLU(0.2),
                                              AdaNorm(args.nz+args.n_label, out_channel, holder=holder))
                else:
                    firstBlock = nn.Sequential() #Dummy

                self.conv = nn.Sequential(firstBlock,
                                          ada_conv2D(out_channel, out_channel,
                                                      kernel2, padding=pad2),
                                          nn.LeakyReLU(0.2),
                                          AdaNorm(args.nz+args.n_label, out_channel, holder=holder))

            elif pixel_norm:
                self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel,
                                                      kernel1, padding=pad1),
                                          PixelNorm(),
                                          nn.LeakyReLU(0.2),
                                          EqualConv2d(out_channel, out_channel,
                                                      kernel2, padding=pad2),
                                          PixelNorm(),
                                          nn.LeakyReLU(0.2))
            else:
                self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel,
                                                      kernel1, padding=pad1),
                                          nn.LeakyReLU(0.2),
                                          nn.Conv2d(out_channel, out_channel,
                                                      kernel2, padding=pad2),
                                          nn.LeakyReLU(0.2))

    def forward(self, input):
        out = self.conv(input)

        return out

gen_spectral_norm = False
debug_ada = False
class Generator(nn.Module):
    def __init__(self, nz, n_label=0, arch=None):
        super().__init__()
        self.nz = nz
        #self.tensor_properties = torch.ones(1).to(device=args.device) #hack
        if n_label > 0:
            self.label_embed = nn.Embedding(n_label, n_label)
            self.label_embed.weight.data.normal_()
        self.code_norm = PixelNorm()

        self.adanorm_blocks = nn.ModuleList()
        #self.z_const = torch.ones(512, 4, 4).to(device=args.device)

        HLM = 1 if arch=='small' else 2 # High-resolution Layer multiplier: Use to make the 64x64+ resolution layers larger by this factor (1 = default Balanced Pioneer)
        progression_raw = [ConvBlock(nz, nz, 4, 3, 3, 1, spectral_norm=gen_spectral_norm, const_layer=True, holder=self),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm, holder=self),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm, holder=self),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm, holder=self),
                                          ConvBlock(nz, int(nz/2)*HLM, 3, 1, spectral_norm=gen_spectral_norm, holder=self),
                                          ConvBlock(int(nz/2)*HLM, int(nz/4)*HLM, 3, 1, spectral_norm=gen_spectral_norm, holder=self),
                                          ConvBlock(int(nz/4)*HLM, int(nz/8)*HLM, 3, 1, spectral_norm=gen_spectral_norm, holder=self),
                                          ConvBlock(int(nz/8)*HLM, int(nz/16)*HLM, 3, 1, spectral_norm=gen_spectral_norm, holder=self),
                                          ConvBlock(int(nz/16)*HLM, int(nz/32)*HLM, 3, 1, spectral_norm=gen_spectral_norm, holder=self)]

        to_rgb_raw = [nn.Conv2d(nz, 3, 1), #Each has 3 out channels and kernel size 1x1!
                                     nn.Conv2d(nz, 3, 1),
                                     nn.Conv2d(nz, 3, 1),
                                     nn.Conv2d(nz, 3, 1),
                                     nn.Conv2d(int(nz/2)*HLM, 3, 1),
                                     nn.Conv2d(int(nz/4)*HLM, 3, 1),
                                     nn.Conv2d(int(nz/8)*HLM, 3, 1),
                                     nn.Conv2d(int(nz/16)*HLM, 3, 1),
                                     nn.Conv2d(int(nz/32)*HLM, 3, 1)]

        noise_raw = [nn.ModuleList([NoiseLayer(nz), NoiseLayer(nz)]),
                                   nn.ModuleList([NoiseLayer(nz), NoiseLayer(nz)]),
                                   nn.ModuleList([NoiseLayer(nz), NoiseLayer(nz)]),
                                   nn.ModuleList([NoiseLayer(nz), NoiseLayer(nz)]),
                                   nn.ModuleList([NoiseLayer(int(nz/2)*HLM), NoiseLayer(int(nz/2)*HLM)]),
                                   nn.ModuleList([NoiseLayer(int(nz/4)*HLM), NoiseLayer(int(nz/4)*HLM)]),
                                   nn.ModuleList([NoiseLayer(int(nz/8)*HLM), NoiseLayer(int(nz/8)*HLM)]),
                                   nn.ModuleList([NoiseLayer(int(nz/16)*HLM), NoiseLayer(int(nz/16)*HLM)]),
                                   nn.ModuleList([NoiseLayer(int(nz/32)*HLM), NoiseLayer(int(nz/32)*HLM)])]

        # The args.flip_invariance_layer (when >=0) relates to driving scale-specific invariances in the weakly supervised case.
        # We disable noise by default if there are support layers. This is only to replicate the approach in Deep Automodulators paper.
        # Feel free to enable it otherwise.
        self.has_noise_layers = args.flip_invariance_layer <= -1

        if args.flip_invariance_layer <= -1:
            self.progression = nn.ModuleList(progression_raw)
            self.to_rgb = nn.ModuleList(to_rgb_raw)
            if self.has_noise_layers:
                self.noise = nn.ModuleList(noise_raw)
            Generator.supportBlockPoints = []
        else:
            self.supportTorgbBlock8 = nn.Conv2d(nz, 3, 1)
            self.supportProgression = ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm, holder=self)
            if self.has_noise_layers:
                self.supportNoise = nn.ModuleList([NoiseLayer(nz), NoiseLayer(nz)])
            # Add support blocks like this and also to the Generator.supportBlockPoints
            #TODO: Support adding multiple layers in the following ModuleList concatenators:
            self.progression = nn.ModuleList(progression_raw[:args.flip_invariance_layer] + [self.supportProgression] + progression_raw[args.flip_invariance_layer:])
            self.to_rgb = nn.ModuleList(to_rgb_raw[:args.flip_invariance_layer] + [self.supportTorgbBlock8] + to_rgb_raw[args.flip_invariance_layer:])
            if self.has_noise_layers:
                self.noise = nn.ModuleList(noise_raw[:args.flip_invariance_layer] + [self.supportNoise] + noise_raw[args.flip_invariance_layer:])
            Generator.supportBlockPoints = [args.flip_invariance_layer] # You can add several layers here as a list, but you need to add the argparser support for that.

        mapping_lrmul = 0.01

        self.use_layer_noise = (not args.no_LN and args.flip_invariance_layer <= -1)

        self.z_preprocess = nn.Sequential(
                    nn.Sequential(equal_lr(nn.Linear(nz, nz), gain=mapping_lrmul), nn.LeakyReLU(0.2)),
                    nn.Sequential(equal_lr(nn.Linear(nz, nz), gain=mapping_lrmul), nn.LeakyReLU(0.2)))

        init_linear(self.z_preprocess[0][0], lr_gain=mapping_lrmul)
        init_linear(self.z_preprocess[1][0], lr_gain=mapping_lrmul)

    @property
    def use_layer_noise(self):
        return self.__use_layer_noise

    @use_layer_noise.setter
    def use_layer_noise(self, use_layer_noise):
        self.__use_layer_noise = use_layer_noise

    def create(self):
        print("::create() n/i")

   # Use alt_mix_z for style mixing so that for z of [B x N] and alt_mix_z of [M x N] and N <= B, the first M entreies of z are (partially) mixed with alt_mix_z
    # Since we assume the input in this case is already fully randomized, we don't need to randomize *which* z entires are mixed. But we do need to randomize the layer ID at which
    # the mixing starts.

    def forward(self, input, label, step=0, alpha=-1, content_input=None, style_layer_begin=0, style_layer_end=-1):
        out_act = lambda x: x

        if style_layer_end == -1: #content_input layer IDs go from 0 to (step). The local numbering is reversed so that, at 128x128 when step=5, id=0 <==> (step-5), id=1 <==> (step-4) etc.
            style_layer_end = step+1

        style_layer_end = min(step+1, style_layer_end)

        if style_layer_begin == -1 or style_layer_begin >= style_layer_end:
            return content_input

        assert(not input is None)

        # Label is reserved for future use. Make None if not in use. #label = self.label_embed(label)
        if not label is None:
            input = torch.cat([input, label], 1)

        batchN = input.size()[0]
        
        if args.stylefc > 0:
            input = self.z_preprocess[0](input)

        for anb in self.adanorm_blocks:
            anb.update(input)

        # For 3 levels of coarseness, for the 2 resnet-block layers, in both the generator and generator_running. Since the layers are just added to the global list without specific indices, the resulting index numbers are ad hoc but atm they are deterministically like here:
        def layers_for_block_depth(d, holder):
             # Generator layers start from 0 and running-Generatro from 17, or vice versa. For both, do the same styling.
            network_offset = int(len(holder.adanorm_blocks) / 2) #17
            return [d*2, d*2+1] #, d*2+network_offset, d*2+network_offset+1]
        
        # The first conv call will start from a constant content_input defined as a class-level var in AdaNorm
        if content_input is None:
            out = torch.ones(512, 4, 4).to(device=input.device).repeat(batchN, 1, 1, 1)
        else:
            out = content_input

        block_offset = 0

        for i in range(style_layer_begin, style_layer_end):
            if i > 0 and not i in Generator.supportBlockPoints:
                if args.upsampling != 'bilinear':
                    upsample = F.upsample(out, scale_factor=2)
                else:
                    upsample = F.interpolate(out, align_corners=False, scale_factor=2, mode='bilinear')
            else:
                upsample = out

            out = upsample

            if i==0 or not self.use_layer_noise:
                out = self.progression[i](out)
            else:
                out = self.progression[i].conv[0][0](out)
                out = self.progression[i].conv[0][1](out)
                out = self.noise[i][0](out)
                out = self.progression[i].conv[0][2](out) #act
                if args.upsampling != 'bilinear':
                    out = self.progression[i].conv[0][3](out) #Blur
                out = self.progression[i].conv[1](out)
                out = self.noise[i][1](out)
                out = self.progression[i].conv[2](out) #act
                out = self.progression[i].conv[3](out)

        if style_layer_end == step+1: # The final layer is ALWAYS either to_rgb layer, or a mixture of 2 to-rgb_layers!
            out = out_act(self.to_rgb[step](out))
            
            if style_layer_end > 1 and 0 <= alpha < 1:
                skip_rgb = out_act(self.to_rgb[step - 1](upsample))
                if args.gnn:
                    channelwise_std = skip_rgb.std((0,2,3), keepdim=True)
                    channelwise_mean = skip_rgb.mean((0,2,3), keepdim=True)

                    out_std = out.std(dim=(0,2,3), keepdim=True)
                    out_mean = out.mean(dim=(0,2,3), keepdim=True)

                    skip_rgb = (skip_rgb - channelwise_mean) * (out_std / channelwise_std) + out_mean

                out = (1 - alpha) * skip_rgb + alpha * out

        return out

pixelNormInDiscriminator = False
use_mean_std_layer = False
spectralNormInDiscriminator = True

class Discriminator(nn.Module):
    def __init__(self, nz, n_label=10, binary_predictor = True):
        super().__init__()
        self.binary_predictor = binary_predictor
        self.progression = nn.ModuleList([ConvBlock(int(nz/32), int(nz/16), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False),
                                          ConvBlock(int(nz/16), int(nz/8), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False),
                                          ConvBlock(int(nz/8), int(nz/4), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False),
                                          ConvBlock(int(nz/4), int(nz/2), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False),
                                          ConvBlock(int(nz/2), nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False),
                                          ConvBlock((nz+1 if use_mean_std_layer else nz), nz, 3, 1, 4, 0,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    ada_norm=False,
                                                    last_act=None)])

        self.from_rgb = nn.ModuleList([nn.Conv2d(3, int(nz/32), 1),
                                       nn.Conv2d(3, int(nz/16), 1),
                                       nn.Conv2d(3, int(nz/8), 1),
                                       nn.Conv2d(3, int(nz/4), 1),
                                       nn.Conv2d(3, int(nz/2), 1),
                                       nn.Conv2d(3, nz, 1),
                                       nn.Conv2d(3, nz, 1),
                                       nn.Conv2d(3, nz, 1),
                                       nn.Conv2d(3, nz, 1)])

        self.n_layer = len(self.progression)

        if self.binary_predictor:
            self.linear = nn.Linear(nz, 1 + n_label)
    c = 0
    def forward(self, input, step, alpha, unused_arg=False):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0 and use_mean_std_layer:
                mean_std = input.std(0).mean()
                mean_std = mean_std.expand(input.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                out = F.avg_pool2d(out, 2)

                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out
        
        z_out = out.squeeze(2).squeeze(2)

        if self.binary_predictor:
            out = self.linear(z_out)
            return out[:, 0], out[:, 1:]
        else:
            out = z_out.view(z_out.size(0), -1)
            ret = utils.normalize(out)

            return  ret
