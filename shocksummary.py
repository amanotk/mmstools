# -*- coding: utf-8 -*-

"""Summary Plot for Bow Shock Crossings with Burst-mode

"""
import os
os.environ['PYTPLOT_NO_GRAPHICS'] = '1'

import numpy as np
import pytplot
import pyspedas
import insitu


def get_data_regexp(d, pattern):
    import re
    pattern = re.compile(pattern)
    found = dict()
    for key in d.keys():
        if re.match(pattern, key):
            found[key] = d[key]
    if len(found) > 1:
        print(d.keys())
        raise ValueError('Error: multiple items found')
    elif len(found) == 0:
        raise ValueError('Error: No items found')
    else:
        key, item = found.popitem()
        return item


def load(probe, trange):
    from pyspedas import mms

    kwargs = {
        'probe'     : probe,
        'trange'    : trange,
        'time_clip' : 1,
        'data_rate' : 'brst',
    }

    # FGM
    varformat = r'*b_gse*'
    mms.fgm(varformat='*b_gse*', **kwargs)
    fgm = get_data_regexp(pytplot.data_quants, r'.*fgm_b_gse.*')

    # SCM
    varformat = r'*acb_gse_scb_*'
    mms.scm(varformat=varformat, **kwargs)
    scm = get_data_regexp(pytplot.data_quants, r'.*acb_gse_scb_.*')

    # EDP
    varformat = r'*dce_gse*'
    mms.edp(varformat=varformat, **kwargs)
    edp = get_data_regexp(pytplot.data_quants, r'.*dce_gse.*')

    # FPI
    varformat = r'*(energyspectr_omni|numberdensity|bulkv_gse)*'
    mms.fpi(datatype='des-moms', varformat=varformat, **kwargs)
    mms.fpi(datatype='dis-moms', varformat=varformat, **kwargs)
    dis_n = get_data_regexp(pytplot.data_quants, r'.*dis_numberdensity_brst.*')
    des_n = get_data_regexp(pytplot.data_quants, r'.*des_numberdensity_brst.*')
    dis_v = get_data_regexp(pytplot.data_quants, r'.*dis_bulkv_gse.*')
    des_v = get_data_regexp(pytplot.data_quants, r'.*des_bulkv_gse.*')
    dis_f = get_data_regexp(pytplot.data_quants, r'.*dis_energyspectr_omni.*')
    des_f = get_data_regexp(pytplot.data_quants, r'.*des_energyspectr_omni.*')

    data = {
        'fgm'   : fgm,
        'scm'   : scm,
        'edp'   : edp,
        'dis_n' : dis_n,
        'des_n' : des_n,
        'dis_v' : dis_v,
        'des_v' : des_v,
        'dis_f' : dis_f,
        'des_f' : des_f,
    }

    return data


def calc_fce(data):
    from insitu import const

    btot = np.linalg.norm(data['fgm'].values[:,0:3], axis=-1)
    fce  = np.abs(const.qme*btot/const.c) / (2*np.pi)

    x = data['fgm'].time.values
    y = np.repeat(fce[:,None], 3, axis=1)
    y[:,1] = y[:,0] * 0.5
    y[:,2] = y[:,0] * 0.1
    fce = insitu.create_xarray(x=x, y=y)
    insitu.set_plot_option(fce,
                           legend=None,
                           linecolor=['w', 'w', 'w'])
    data['fce'] = fce

    return data

def calc_wavepower(data):
    from insitu import wave

    scm = data['scm']
    edp = data['edp']
    fs  = 8192
    ns  = 1024
    win = 'blackman'

    data['scm_spec'] = wave.spectrogram([scm[:,0], scm[:,1], scm[:,2]],
                                        fs, ns, window=win)
    data['edp_spec'] = wave.spectrogram([edp[:,0], edp[:,1], edp[:,2]],
                                        fs, ns, window=win)

    return data


def set_plot_options(data):
    if 'fgm' in data:
        insitu.set_plot_option(data['fgm'],
                               ylabel='B [nT]')
    if 'dis_n' in data:
        insitu.set_plot_option(data['dis_n'],
                               ylabel='Density [1/cm^3]',
                               legend=['Ni'],
                               linecolor='r')
    if 'des_n' in data:
        insitu.set_plot_option(data['des_n'],
                               ylabel='Density [1/cm^3]',
                               legend=['Ne'],
                               linecolor='b')
    if 'dis_v' in data:
        insitu.set_plot_option(data['dis_v'],
                               ylabel='Velocity [km/s]',
                               legend=['Vix', 'Viy', 'Viz'])
    if 'des_v' in data:
        insitu.set_plot_option(data['des_v'],
                               ylabel='Velocity [km/s]',
                               legend=['Vex', 'Vey', 'Vez'])
    if 'dis_f' in data:
        zlabel = insitu.get_plot_option(data['dis_f'], 'zlabel')
        insitu.set_plot_option(data['dis_f'],
                               ylabel='DIS (omni) [eV]',
                               zlabel=zlabel,
                               colormap='viridis')
    if 'des_f' in data:
        zlabel = insitu.get_plot_option(data['des_f'], 'zlabel')
        insitu.set_plot_option(data['des_f'],
                               ylabel='DES (omni) [eV]',
                               zlabel=zlabel,
                               colormap='viridis')
    if 'scm_spec' in data:
        insitu.set_plot_option(data['scm_spec'],
                               ylabel='Frequency [Hz]',
                               yrange=[1.0e+1, 1.0e+3],
                               zlabel='[nT^2 / Hz]',
                               zrange=[-7, 2])
    if 'edp_spec' in data:
        insitu.set_plot_option(data['edp_spec'],
                               ylabel='Frequency [Hz]',
                               yrange=[1.0e+1, 1.0e+3],
                               zlabel='[mV^2 / Hz]',
                               zrange=[-7, 2])


def plot(data, **kwargs):
    data = calc_wavepower(data)
    data = calc_fce(data)
    set_plot_options(data)
    items = [
        data['fgm'],
        [data['dis_n'], data['des_n']],
        data['dis_v'],
        data['dis_f'],
        data['des_f'],
        data['scm_spec'],
        data['edp_spec'],
        #[data['scm_spec'], data['fce']],
        #[data['edp_spec'], data['fce']],
    ]

    if not 'width' in kwargs:
        kwargs['width'] = 800
    if not 'height' in kwargs:
        kwargs['height'] = 1200

    return insitu.tplot(items, **kwargs)


def load_and_plot(probe, trange, **kwargs):
    data = load(probe, trange)
    figure = plot(data, **kwargs)
    return figure, data
