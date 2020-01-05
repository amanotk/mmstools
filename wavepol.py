#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Wave Polarization Analysis

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


def get_extended_trange(trange, delt):
    t1 = insitu.to_datestring(insitu.to_unixtime(trange[0]) - delt)
    t2 = insitu.to_datestring(insitu.to_unixtime(trange[1]) + delt)
    return [t1, t2]


def load(probe, trange):
    from pyspedas import mms

    kwargs = {
        'probe'     : probe,
        'trange'    : get_extended_trange(trange, 10.0),
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


def calc_wavepol(data, ns):
    from insitu import wave

    fgm = data['fgm']
    scm = data['scm']
    edp = data['edp']
    fs  = 8192
    win = 'blackman'

    data['scm_spec'] = wave.spectrogram([scm[:,0], scm[:,1], scm[:,2]], fs,
                                        nperseg=ns, noverlap=ns//2, window=win)

    svd = wave.SVD(nperseg=ns, noverlap=ns//2, window=win)
    result = svd.analyze(edp, scm, fgm)

    var_names = ('psd', 'planarity', 'degpol', 'ellipticity',
                 'theta_kb', 'theta_sb',)
    for v in var_names:
        data[v] = result[v]

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
    if 'fce' in data:
        insitu.set_plot_option(data['fce'],
                               legend=None,
                               linecolor=['k', 'k', 'k'])


def plot(data, trange, **kwargs):
    ns = 512
    na = 16
    data = calc_wavepol(data, ns)
    data = calc_fce(data, na)

    # set plot options
    set_plot_options(data)

    print(insitu.to_datestring(data['scm_spec'].time[:5], '%M:%S.%f'))
    print(insitu.to_datestring(data['psd'].time[:5], '%M:%S.%f'))
    print(insitu.to_datestring(data['scm_spec'].time[-5:], '%M:%S.%f'))
    print(insitu.to_datestring(data['psd'].time[-5:], '%M:%S.%f'))
    print(data['scm_spec'].time.shape)
    print(data['psd'].time.shape)

    data['psd']['time'] = data['scm_spec']['time']

    items = [
        data['fgm'],
        data['dis_v'],
        [data['scm_spec'], data['fce']],
        [data['psd'], data['fce']],
        [data['degpol'], data['fce']],
        [data['ellipticity'], data['fce']],
        [data['theta_kb'], data['fce']],
        [data['theta_sb'], data['fce']],
    ]

    if not 'width' in kwargs:
        kwargs['width'] = 800
    if not 'height' in kwargs:
        kwargs['height'] = 800

    return insitu.tplot(items, trange=trange, **kwargs)


def load_and_plot(probe, trange, **kwargs):
    kwargs['title'] = 'MMS%d ' % (probe) + \
                      insitu.to_datestring(trange[0], '%Y-%m-%d %H:%M:%S')
    data = load(probe, trange)
    figure = plot(data, trange, **kwargs)
    return figure, data


if __name__ == '__main__':
    import argparse
    description = """Generate Plot for Wave Polarization Analysis"""
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
