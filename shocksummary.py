#!/usr/bin/env python
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


def calc_fce(data, na):
    from insitu import const

    btot = np.linalg.norm(data['fgm'].values[:,0:3], axis=-1) * 1.0e-9
    fce  = np.abs(const.qme*btot) / (2*np.pi)

    x = data['fgm'].time.values
    y = np.repeat(fce[:,None], 3, axis=1)
    y[:,1] = y[:,0] * 0.5
    y[:,2] = y[:,0] * 0.1
    fce = insitu.create_xarray(x=x, y=y)
    data['fce'] = fce.rolling(time=na, center='True').mean()
    data['fce'].attrs = fce.attrs

    return data


def calc_wavepower(data, ns):
    from insitu import wave

    scm = data['scm']
    edp = data['edp']
    fs  = 8192
    win = 'blackman'

    data['scm_spec'] = wave.spectrogram([scm[:,0], scm[:,1], scm[:,2]],
                                        fs, ns, window=win)
    data['edp_spec'] = wave.spectrogram([edp[:,0], edp[:,1], edp[:,2]],
                                        fs, ns, window=win)

    return data


def set_plot_options(data, colormap='viridis'):
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
                               zrange=[3, 9],
                               colormap=colormap)
    if 'des_f' in data:
        zlabel = insitu.get_plot_option(data['des_f'], 'zlabel')
        insitu.set_plot_option(data['des_f'],
                               ylabel='DES (omni) [eV]',
                               zlabel=zlabel,
                               zrange=[4, 10],
                               colormap=colormap)
    if 'scm_spec' in data:
        insitu.set_plot_option(data['scm_spec'],
                               ylabel='Frequency [Hz]',
                               yrange=[1.0e+1, 4.0e+3],
                               zlabel='[nT^2 / Hz]',
                               zrange=[-8, 0],
                               colormap=colormap)
        spec = data['scm_spec'].values
        data['scm_spec'].values = np.where(spec > 1.0e-8, spec, None)

    if 'edp_spec' in data:
        insitu.set_plot_option(data['edp_spec'],
                               ylabel='Frequency [Hz]',
                               yrange=[1.0e+1, 4.0e+3],
                               zlabel='[mV^2 / Hz]',
                               zrange=[-7, 1],
                               colormap=colormap)
        spec = data['edp_spec'].values
        data['edp_spec'].values = np.where(spec > 1.0e-7, spec, None)
    if 'fce' in data:
        insitu.set_plot_option(data['fce'],
                               legend=None,
                               linecolor=['k', 'k', 'k'])
        data['fce'].values = np.log10(data['fce'].values)


def plot(data, **kwargs):
    ns = 1024
    na = 16
    data = calc_wavepower(data, ns)
    data = calc_fce(data, na)

    # set plot options
    set_plot_options(data, 'jet')

    items = [
        data['fgm'],
        [data['dis_n'], data['des_n']],
        data['dis_v'],
        data['dis_f'],
        data['des_f'],
        [data['scm_spec'], data['fce']],
        [data['edp_spec'], data['fce']],
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


if __name__ == '__main__':
    import argparse
    description = """Generate Summary Plot for Bow Shock Crossings"""
    parser = argparse.ArgumentParser(description=description)

    # add command line options
    parser.add_argument('-t', '--trange',
                        dest='trange',
                        nargs=2,
                        type=str,
                        required=True,
                        help='time interval')
    parser.add_argument('-o', '--output',
                        dest='output',
                        type=str,
                        required=True,
                        help='output filename')
    parser.add_argument('-p', '--probe',
                        dest='probe',
                        type=int,
                        default=1,
                        help='spacecraft ID')

    # parse
    args = parser.parse_args()
    output = args.output
    trange = args.trange
    probe  = args.probe
    if probe < 1 or probe > 4:
        print('No such spacecraft ID : %d --- use MMS1 instead' % (probe))
        probe = 1

    figure, data = load_and_plot(probe, trange, backend='mpl')
    figure.savefig(output)
