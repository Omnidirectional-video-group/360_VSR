"""
Original Author: 
Mediatek NeuroPilot Ecosystem
    
    Original Source: https://github.com/MediaTek-NeuroPilot/mai22-real-time-video-sr
"""

import importlib.util
import pathlib

def plugin_from_file(location, name, base):
    spec = importlib.util.spec_from_file_location(pathlib.PurePath(location).stem, location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        plugin = getattr(module, name)
    except AttributeError:
        raise AttributeError(f'Given plugin name {name} is not found.')



    return plugin