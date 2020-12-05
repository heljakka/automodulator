dependencies = ['torch']

import torch
import pioneer.session

def ffhq512(pretrained, **kwargs):
    """ The Automodulator FFHQ 512x512 pre-trained model
    """
    if pretrained: 
        checkpoint_index = 42420000
        save_dir = 'https://zenodo.org/record/4298894/files/42420000_state.pth?download=1'
    else:
        checkpoint_index = 0
        save_dir = 'ffhq512'

    return pioneer.session.Session( pretrained=pretrained, save_dir=save_dir, \
                                    transform_key='ffhq512', start_iteration=checkpoint_index, **kwargs)

def celebahq256(pretrained, **kwargs):
    """ The Automodulator CelebA-HQ 256x256 pre-trained model
    """
    if pretrained: 
        checkpoint_index = 31877148
        save_dir = 'https://zenodo.org/record/4298894/files/31877148_state.pth?download=1'
    else:
        checkpoint_index = 0
        save_dir = 'celebaHQ256'

    return pioneer.session.Session( pretrained=pretrained, save_dir=save_dir, \
                                    start_iteration=checkpoint_index, arch='small', **kwargs)

def ffhq256(pretrained, **kwargs):
    """ The Automodulator FFHQ 256x256 pre-trained model
    """
    if pretrained: 
        checkpoint_index = 34740000
        save_dir = 'https://zenodo.org/record/4298894/files/34740000_state.pth?download=1'
    else:
        checkpoint_index = 0
        save_dir = 'ffhq256'

    return pioneer.session.Session( pretrained=pretrained, save_dir=save_dir, \
                                    start_iteration=checkpoint_index, **kwargs)

def lsuncars256(pretrained, **kwargs):
    """ The Automodulator LSUN Cars 256x256 pre-trained model
    """

    if pretrained: 
        checkpoint_index = 32057140
        save_dir = 'https://zenodo.org/record/4298894/files/32057140_state.pth?download=1'
    else:
        checkpoint_index = 0
        save_dir='lsunCars256'

    return pioneer.session.Session( pretrained=pretrained, save_dir=save_dir, \
                                    start_iteration=checkpoint_index, **kwargs)

def lsunbedrooms256(pretrained, **kwargs):
    """ The Automodulator LSUN Bedrooms 256x256 pre-trained model
    """
    if pretrained: 
        checkpoint_index = 27574284
        save_dir = 'https://zenodo.org/record/4298894/files/27574284_state.pth?download=1'
    else:
        checkpoint_index = 0
        save_dir='lsunBedrooms256'

    return pioneer.session.Session( pretrained=pretrained, save_dir=save_dir, \
                                    start_iteration=checkpoint_index, **kwargs)
