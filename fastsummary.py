#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Summary Plot for Fast Survey

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
    }

    # FGM
    varformat = r'*b_gse*'
    mms.fgm(varformat='*b_gse*', data_rate='srvy', **kwargs)
    fgm = get_data_regexp(pytplot.data_quants, r'.*fgm_b_gse.*')

    # FPI
    varformat = r'*(energyspectr_omni|numberdensity|bulkv_gse)*'
    mms.fpi(data_rate='fast', datatype='des-moms', varformat=varformat, **kwargs)
    mms.fpi(data_rate='fast', datatype='dis-moms', varformat=varformat, **kwargs)
    dis_n = get_data_regexp(pytplot.data_quants, r'.*dis_numberdensity_fast.*')
    des_n = get_data_regexp(pytplot.data_quants, r'.*des_numberdensity_fast.*')
    dis_v = get_data_regexp(pytplot.data_quants, r'.*dis_bulkv_gse.*')
    des_v = get_data_regexp(pytplot.data_quants, r'.*des_bulkv_gse.*')
    dis_f = get_data_regexp(pytplot.data_quants, r'.*dis_energyspectr_omni.*')
    des_f = get_data_regexp(pytplot.data_quants, r'.*des_energyspectr_omni.*')

    data = {
        'fgm'   : fgm,
        'dis_n' : dis_n,
        'des_n' : des_n,
        'dis_v' : dis_v,
        'des_v' : des_v,
        'dis_f' : dis_f,
        'des_f' : des_f,
    }

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


def plot(data, **kwargs):
    # set plot options
    set_plot_options(data, 'viridis')

    items = [
        data['fgm'],
        [data['dis_n'], data['des_n']],
        data['dis_v'],
        data['dis_f'],
        data['des_f'],
    ]

    if not 'width' in kwargs:
        kwargs['width'] = 800
    if not 'height' in kwargs:
        kwargs['height'] = 800

    return insitu.tplot(items, **kwargs)


def load_and_plot(probe, trange, **kwargs):
    data = load(probe, trange)
    figure = plot(data, trange=trange, **kwargs)
    return figure, data


if __name__ == '__main__':
    import argparse
    description = """Generate Summary Plot for Fast Survey"""
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
