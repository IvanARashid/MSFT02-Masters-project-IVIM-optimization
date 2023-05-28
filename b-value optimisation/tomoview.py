# -*- coding: utf-8 -*-
"""
Created on Thu Jul 1 10:39:00 2021

Functions to view volume data and change plotting behaviour

@author: Till, till.dreier@med.lu.se
"""

import matplotlib.pyplot as plt
from IPython import get_ipython
import numpy as np


def __arrow_navigation__(event, z, Z):
    if event.key == "up":
        z = min(z + 1, Z - 1)
    elif event.key == 'down':
        z = max(z - 1, 0)
    elif event.key == 'right':
        z = min(z + 10, Z - 1)
    elif event.key == 'left':
        z = max(z - 10, 0)
    elif event.key == 'pagedown':
        z = min(z + 50, Z + 1)
    elif event.key == 'pageup':
        z = max(z - 50, 0)
    return z


def view_volume(vol, figure_num=1, cmap='gray', vmin=None, vmax=None):
    """
    Shows volumetric data for interactive inspection.
    Left/Right keys :   ± 10 projections
    Up/Down keys:       ± 1 projection
    Page Up/Down keys:  ± 50 projections
    Should work in Spyder. In PyCharm, change the plotting backend (see test script for details).
    """

    def update_drawing():
        ax.images[0].set_array(vol[z])
        ax.set_title('slice {}/{}'.format(z, vol.shape[0]))
        fig.canvas.draw()

    def key_press(event):
        nonlocal z
        z = __arrow_navigation__(event, z, Z)
        update_drawing()

    Z = vol.shape[0]
    z = (Z - 1) // 2
    fig, ax = plt.subplots(num=figure_num, dpi=200)
    if vmin is None:
        vmin = np.min(vol)
    if vmax is None:
        vmax = np.max(vol)
    ax.imshow(vol[z], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title('slice {}/{}'.format(z, vol.shape[0]))
    fig.canvas.mpl_connect('key_press_event', key_press)


def spyder_inline_plot():
    """
    In Spyder editor, switch to inline plotting.
    Set plotting backend to 'automatic' in: Tools >> Preferences >> IPython console >> Graphics
    """
    # FIXME: check if this works on macos as well
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except:
        print('Could not set inline plotting')


def spyder_window_plot(os='windows'):
    """
    In Spyder editor, switch to plotting in a window, allowing interactive plots.
    Set plotting backend to 'automatic' in: Tools >> Preferences >> IPython console >> Graphics
    """
    if os.lower() in ['windows', 'win', 'linux']:
        try:
            get_ipython().run_line_magic('matplotlib', 'qt5')
        except:
            print('Could not update plotting backend to Qt5Agg for Windows or Linux OS.')
    elif os.lower() in ['mac', 'macos', 'macosx']:
        try:
            get_ipython().run_line_magic('matplotlib', 'macosx')
        except:
            print('Could not set interactive macosx plotting backend.')
    else:
        print('Invlid OS, cannot set plotting backend.')