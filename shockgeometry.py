#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shock Gemetry Analysis Tool

"""
import os
import sys
import warnings
import json

import numpy as np
import scipy as sp
from scipy import optimize
from scipy import signal
from scipy import constants
import pandas as pd

# matplotlib
import matplotlib as mpl
from matplotlib import pylab as plt

import aspy
const = aspy.const

import pyspedas
import pytplot

DIR_FMT = '%Y%m%d_%H%M%S'


class MixedVMCAnalyzer:
    def __init__(self, window, deltat):
        if type(window) == int and type(deltat) == np.timedelta64:
            self.window = window
            self.deltat = deltat
        else:
            raise ValueError('Invalid input')

    def select_interval(self, t1, t2, U, B):
        index1 = U.time.searchsorted(t1.to_numpy())
        index2 = U.time.searchsorted(t2.to_numpy())
        return index1, index2, U.values[index1:index2+1], B.values[index1:index2+1]

    def make_pairs(self, U1, U2, B1, B2):
        N1 = U1.shape[0]
        N2 = U2.shape[0]
        U1, U2 = np.broadcast_arrays(U1[:,None,:], U2[None,:,:])
        B1, B2 = np.broadcast_arrays(B1[:,None,:], B2[None,:,:])
        U1 = U1.reshape((N1*N2, 3))    
        U2 = U2.reshape((N1*N2, 3))    
        B1 = B1.reshape((N1*N2, 3))    
        B2 = B2.reshape((N1*N2, 3))    
        return U1, U2, B1, B2

    def get_shock_parameters(self, U1, U2, B1, B2):
        U1, U2, B1, B2 = self.make_pairs(U1, U2, B1, B2)
        E1 = np.cross(-U1, B1, axis=-1)
        E2 = np.cross(-U2, B2, axis=-1)
        dB = B2 - B1
        dU = U2 - U1

        # LMN coordinate
        nvec = np.cross(np.cross(dB, dU, axis=-1), dB, axis=-1)
        nvec = nvec * np.sign(nvec[0] + nvec[1] + nvec[2])
        nvec = nvec / np.linalg.norm(nvec, axis=-1)[:,None]
        lvec = dB - np.sum(dB*nvec, axis=-1)[:,None] * nvec
        lvec = lvec * np.sign(np.sum(B1*lvec, axis=-1))[:,None]
        lvec = lvec / np.linalg.norm(lvec, axis=-1)[:,None]
        mvec = np.cross(nvec, lvec, axis=-1)
        mvec = mvec / np.linalg.norm(mvec, axis=-1)[:,None]

        # Bl : difference of tangential component of B-field
        # Bn : difference of normal component of B-field
        # Em : difference of out-of-coplanarity component of E-field
        # Vs : shock speed in s/c frame 
        Bl = np.sum(dB*lvec, axis=-1)
        Bn = np.sum(dB*nvec, axis=-1)
        Em = np.sum((E2 - E1)*mvec, axis=-1)
        Vs = - Em / Bl

        return lvec, mvec, nvec, Vs, (U1, U2, B1, B2)

    def get_best_interval(self, U1, U2, B1, B2):
        N1 = U1.shape[0]
        N2 = U2.shape[0]
        lvec, mvec, nvec, Vs, (U1, U2, B1, B2) = self.get_shock_parameters(U1, U2, B1, B2)

        E1 = np.cross(-U1, B1, axis=-1)
        E2 = np.cross(-U2, B2, axis=-1)
        Bn = np.sum((B2 - B1)*nvec, axis=-1)
        Em = np.sum((E2 - E1)*mvec, axis=-1)

        # window
        W = self.window
        w = np.ones(2*W + 1)
        w = w[:,None] * w[None,:] / (np.sum(w) * np.sum(w))

        # mask
        mask1 = np.logical_or(np.less(np.arange(N1), W), np.greater_equal(np.arange(N1), N1-W))
        mask2 = np.logical_or(np.less(np.arange(N2), W), np.greater_equal(np.arange(N2), N2-W))
        mask = np.logical_or(mask1[:,None], mask2[None,:])

        # moving average
        bn0 = Bn.reshape((N1, N2))
        bn1 = signal.convolve2d(bn0, w, mode='same')
        chi_bn = bn1**2 / bn0.mean()**2

        em0 = Em.reshape((N1, N2))
        em1 = signal.convolve2d(em0, w, mode='same')
        chi_em = em1**2 / em0.mean()**2

        # find index with the minimum error
        err_bn = np.ma.masked_array(chi_bn, mask=mask)
        err_em = np.ma.masked_array(chi_em, mask=mask)
        index1, index2 = np.unravel_index(np.argmin(err_bn + err_em), (N1, N2))

        lvec = lvec.reshape((N1, N2, 3))[index1,index2]
        mvec = mvec.reshape((N1, N2, 3))[index1,index2]
        nvec = nvec.reshape((N1, N2, 3))[index1,index2]
        Vs = Vs.reshape((N1, N2))[index1,index2]

        return index1, index2, lvec, mvec, nvec, Vs

    def estimate_error(self, U1, U2, B1, B2):
        lvec, mvec, nvec, Vs, _ = self.get_shock_parameters(U1, U2, B1, B2)

        _, theta, phi = aspy.xyz2sph(nvec[:,0], nvec[:,1], nvec[:,2])
        error_nvec = np.sqrt(0.5*(np.var(theta) + np.var(phi)))
        error_vshn = np.std(Vs)
        return error_nvec, error_vshn

    def __call__(self, trange, data_dict, dirname):
        t1, t2 = trange
        U = data_dict['vi']
        B = data_dict['bf']
        T = 0.5*(t2 - t1) - np.timedelta64(10, 's')

        # candidate left interval
        tl1 = pd.to_datetime(t1) - self.deltat
        tl2 = pd.to_datetime(t1) + T
        il1, il2, Ul, Bl = self.select_interval(tl1, tl2, U, B)

        # candidate right interval
        tr1 = pd.to_datetime(t2) - T
        tr2 = pd.to_datetime(t2) + self.deltat
        ir1, ir2, Ur, Br = self.select_interval(tr1, tr2, U, B)

        # find best interval
        index1, index2, lvec, mvec, nvec, Vs = self.get_best_interval(Ul, Ur, Bl, Br)
        l_index = index1 - self.window + il1, index1 + self.window + il1
        r_index = index2 - self.window + ir1, index2 + self.window + ir1
        l_trange = B.time.values[l_index[0]], B.time.values[l_index[1]]
        r_trange = B.time.values[r_index[0]], B.time.values[r_index[1]]

        ul = U.values[l_index[0]:l_index[1]+1,:]
        bl = B.values[l_index[0]:l_index[1]+1,:]
        ur = U.values[r_index[0]:r_index[1]+1,:]
        br = B.values[r_index[0]:r_index[1]+1,:]

        # estiamte errors
        error_nvec, error_vshn = self.estimate_error(ul, ur, bl, br)

        # transformation velocity to NIF
        V = 0.5*(ul.mean(axis=0) + ur.mean(axis=0))
        Vshock = np.array([np.dot(V, lvec), np.dot(V, mvec), Vs])

        # store result
        result = dict(
            l_index=l_index,
            r_index=r_index,
            l_trange=l_trange,
            r_trange=r_trange,
            lvec=lvec,
            mvec=mvec,
            nvec=nvec,
            Vshock=Vshock,
            error_nvec=error_nvec,
            error_vshn=error_vshn,
        )

        # summary plot with LMN coordinate in NIF for visual inspection
        t1 = pd.to_datetime(tl1) - np.timedelta64(1, 'm')
        t2 = pd.to_datetime(tr2) + np.timedelta64(1, 'm')
        plot_summary_lmn([t1, t2], data_dict, result, dirname)

        # calculate and save parameters
        result = save_parameters(data_dict, result, dirname)

        return result


def plot_summary_lmn(trange, data_dict, result, dirname):
    lvec = result['lvec']
    mvec = result['mvec']
    nvec = result['nvec']
    LMN  = np.vstack([lvec, mvec, nvec])[None,:,:]
    Vshock = result['Vshock']

    ne = data_dict['ne'].values
    ni = data_dict['ni'].values
    bf = data_dict['bf'].values
    vi = data_dict['vi'].values
    tt = data_dict['bf'].time.values

    # convert to LMN coordinate in NIF
    bf_lmn = np.sum(LMN*bf[:,None,:], axis=-1)
    vi_lmn = np.sum(LMN*vi[:,None,:], axis=-1) - Vshock[None,:]
    ef_lmn = np.cross(bf_lmn, vi_lmn, axis=-1) * 1.0e-3

    # store data to plot
    density = np.vstack([ni, ne]).swapaxes(0, 1)
    pytplot.store_data('density', data=dict(x=tt, y=density))
    pytplot.store_data('bf_lmn', data=dict(x=tt, y=bf_lmn))
    pytplot.store_data('ef_lmn', data=dict(x=tt, y=ef_lmn))
    pytplot.store_data('vi_lmn', data=dict(x=tt, y=vi_lmn))

    # set plot options
    aspy.set_plot_option(pytplot.data_quants['density'],
                         ylabel=r'N [1/cm$^3$]',
                         legend=('Ni', 'Ne'),
                         line_color=('r', 'b'),
                         char_size=10)
    aspy.set_plot_option(pytplot.data_quants['bf_lmn'],
                         ylabel='B [nT]',
                         legend=('L', 'M', 'N'),
                         line_color=('b', 'g', 'r'),
                         char_size=10)
    aspy.set_plot_option(pytplot.data_quants['ef_lmn'],
                         ylabel='E [mV/m]',
                         legend=('L', 'M', 'N'),
                         line_color=('b', 'g', 'r'),
                         char_size=10)
    aspy.set_plot_option(pytplot.data_quants['vi_lmn'],
                         ylabel='V [km/s]',
                         legend=('L', 'M', 'N'),
                         line_color=('b', 'g', 'r'),
                         char_size=10)

    # suppress UserWarning in agg backend
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pytplot.tlimit([t.strftime('%Y-%m-%d %H:%M:%S') for t in trange])
        pytplot.tplot_options('axis_font_size', 10)
        pytplot.tplot(['density', 'bf_lmn', 'vi_lmn', 'ef_lmn'], xsize=8, ysize=8)

    # customize appearance
    tl = np.logical_and(tt >= result['l_trange'][0], tt <= result['l_trange'][1])
    tr = np.logical_and(tt >= result['r_trange'][0], tt <= result['r_trange'][1])
    tc = result['l_trange'][1] + 0.5*(result['r_trange'][0] - result['l_trange'][1])
    title = pd.to_datetime(tc).strftime('MMS Bow Shock at %Y-%m-%d %H:%M:%S (Normal Incidence Frame)')
    fig = plt.gcf()
    fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.15)
    fig.suptitle(title, fontsize=10)

    axs = fig.get_axes()
    for ax in axs:
        plt.sca(ax)
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
        plt.grid(True, linestyle='--')
        plt.fill_between(tt, 0, 1, where=tl, color='grey', alpha=0.25, transform=ax.get_xaxis_transform())
        plt.fill_between(tt, 0, 1, where=tr, color='grey', alpha=0.25, transform=ax.get_xaxis_transform())
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mpl.dates.MinuteLocator())
        ax.xaxis.set_minor_locator(mpl.dates.SecondLocator(bysecond=range(0, 60, 10)))
    ax.set_xlabel('UT')

    # save file
    fig.savefig(os.sep.join([dirname, 'summary_lmn_nif.png']))


def save_parameters(data_dict, result, dirname):
    Ne = data_dict['ne']
    Ni = data_dict['ne']
    Ue = data_dict['ve']
    Ui = data_dict['vi']
    Pe = data_dict['pe']
    Pi = data_dict['pi']
    Bf = data_dict['bf']

    l_index = result['l_index']
    r_index = result['r_index']
    lvec = result['lvec']
    mvec = result['mvec']
    nvec = result['nvec']
    Vshock = result['Vshock']

    # upstream and downstream indices
    index1 = np.arange(l_index[0], l_index[1] + 1)
    index2 = np.arange(r_index[0], r_index[1] + 1)
    trange1 = np.datetime_as_string(result['l_trange'])
    trange2 = np.datetime_as_string(result['r_trange'])

    # swap for outbound crossing
    if Ne.values[index1].mean() > Ne.values[index2].mean():
        index1, index2 = index2, index1
        trange1, trange2 = trange1, trange2

    ## take average and standard deviation
    stdmean = lambda f: (np.mean(f, axis=0), np.std(f, axis=0),)

    # upstream
    Ne1_avg, Ne1_err = stdmean(Ne.values[index1])
    Ni1_avg, Ni1_err = stdmean(Ni.values[index1])
    Ue1_avg, Ue1_err = stdmean(Ue.values[index1,:])
    Ui1_avg, Ui1_err = stdmean(Ui.values[index1,:])
    Pe1_avg, Pe1_err = stdmean(Pe.values[index1])
    Pi1_avg, Pi1_err = stdmean(Pi.values[index1])
    Bf1_avg, Bf1_err = stdmean(Bf.values[index1,:])
    # downstream
    Ne2_avg, Ne2_err = stdmean(Ne.values[index2])
    Ni2_avg, Ni2_err = stdmean(Ni.values[index2])
    Ue2_avg, Ue2_err = stdmean(Ue.values[index2,:])
    Ui2_avg, Ui2_err = stdmean(Ui.values[index2,:])
    Pe2_avg, Pe2_err = stdmean(Pe.values[index2])
    Pi2_avg, Pi2_err = stdmean(Pi.values[index2])
    Bf2_avg, Bf2_err = stdmean(Bf.values[index2,:])

    Un1 = np.dot(Ui1_avg, nvec)
    Un2 = np.dot(Ui2_avg, nvec)
    Bn1 = np.dot(Bf1_avg, nvec)
    Bl1 = np.dot(Bf1_avg, lvec)
    Bt1 = np.linalg.norm(Bf1_avg)
    Bt1_err = np.sqrt(Bf1_err[0]**2 + Bf1_err[1]**2 + Bf1_err[2]**2)

    # shock obliquity
    theta_bn = np.rad2deg(np.arctan2(Bl1, Bn1))
    theta_bn_err = result['error_nvec']

    # shock speed
    Vs_n_scf = np.abs(Vshock[2])
    Vs_n_scf_err = result['error_vshn']
    Vs_n_nif = np.abs(Un1 - Vs_n_scf)
    Vs_n_nif_err = Vs_n_scf_err

    # Mach number
    Ne1_crct = Ne2_avg * (Un2 - Vshock[2]) / (Un1 - Vshock[2])
    Ne1_crct_err = Ne2_err * (Un2 - Vshock[2]) / (Un1 - Vshock[2])
    Va1_crct = 21.806 * Bt1 / np.sqrt(Ne1_crct)
    Va1_crct_err = Va1_crct * np.sqrt((Bt1_err/Bt1)**2 + 0.25*(Ne1_crct_err/Ne1_crct)**2)
    Ma_n_nif = Vs_n_nif / Va1_crct
    Ma_n_nif_err = Vs_n_nif_err / Va1_crct

    # pressure plasma beta
    Pb  = (np.linalg.norm(Bf1_avg)**2 / (2*constants.mu_0)) * 1e-9

    # average OMNI data
    omni_time_range = slice(
        result['l_trange'][0] - np.timedelta64(30, 'm'),
        result['r_trange'][1] + np.timedelta64(30, 'm')
    )
    omni_stdmean = lambda f: \
        (f.sel(time=omni_time_range).mean().item(), f.sel(time=omni_time_range).std().item())
    Ni_omni = pytplot.get_data('omni_ni', xarray=True)
    Ma_omni = pytplot.get_data('omni_mach', xarray=True)
    Beta_omni = pytplot.get_data('omni_beta', xarray=True)

    parameters = {
        'trange1' : list(trange1),
        'trange2' : list(trange2),
        # upstream
        'Ne1' : (Ne1_avg, Ne1_err),
        'Ni1' : (Ni1_avg, Ni1_err),
        'Uex1' : (Ue1_avg[0], Ue1_err[0]),
        'Uey1' : (Ue1_avg[1], Ue1_err[1]),
        'Uez1' : (Ue1_avg[2], Ue1_err[2]),
        'Uix1' : (Ui1_avg[0], Ui1_err[0]),
        'Uiy1' : (Ui1_avg[1], Ui1_err[1]),
        'Uiz1' : (Ui1_avg[2], Ui1_err[2]),
        'Pe1' : (Pe1_avg, Pe1_err),
        'Pi1' : (Pi1_avg, Pi1_err),
        'Bx1' : (Bf1_avg[0], Bf1_err[0]),
        'By1' : (Bf1_avg[1], Bf1_err[1]),
        'Bz1' : (Bf1_avg[2], Bf1_err[2]),
        # downstream
        'Ne2' : (Ne2_avg, Ne2_err),
        'Ni2' : (Ni2_avg, Ni2_err),
        'Uex2' : (Ue2_avg[0], Ue2_err[0]),
        'Uey2' : (Ue2_avg[1], Ue2_err[1]),
        'Uez2' : (Ue2_avg[2], Ue2_err[2]),
        'Uix2' : (Ui2_avg[0], Ui2_err[0]),
        'Uiy2' : (Ui2_avg[1], Ui2_err[1]),
        'Uiz2' : (Ui2_avg[2], Ui2_err[2]),
        'Pe2' : (Pe2_avg, Pe2_err),
        'Pi2' : (Pi2_avg, Pi2_err),
        'Bx2' : (Bf2_avg[0], Bf2_err[0]),
        'By2' : (Bf2_avg[1], Bf2_err[1]),
        'Bz2' : (Bf2_avg[2], Bf2_err[2]),
        # shock parameters
        'Bt1' : (Bt1, Bt1_err),
        'Ne1_crct' : (Ne1_crct, Ne1_crct_err),
        'Va1_crct' : (Va1_crct, Va1_crct_err),
        'theta_bn' : (theta_bn, theta_bn_err),
        'Vs_n_nif' : (Vs_n_nif, Vs_n_nif_err),
        'Vs_n_scf' : (Vs_n_scf, Vs_n_scf_err),
        'Ma_n_nif' : (Ma_n_nif, Ma_n_nif_err),
        'Beta_i' : (Pi1_avg / Pb, Pi1_err / Pb),
        'Beta_e' : (Pe1_avg / Pb, Pe1_err / Pb),
        # coordinate
        'lvec' : list(lvec),
        'mvec' : list(mvec),
        'nvec' : list(nvec),
        # OMNI
        'Ni_omni' : omni_stdmean(Ni_omni),
        'Ma_omni' : omni_stdmean(Ma_omni),
        'Beta_omni' : omni_stdmean(Beta_omni),
    }

    print('{:20s} : {}'.format('trange1', parameters['trange1']))
    print('{:20s} : {}'.format('trange2', parameters['trange2']))
    keywords = ('Ne1', 'Ne1_crct', 'Bt1', 'Va1_crct', 'theta_bn',
        'Vs_n_scf', 'Vs_n_nif', 'Ma_n_nif', 'Beta_i', 'Beta_e',
        'Ni_omni', 'Ma_omni', 'Beta_omni')
    for key in keywords:
        if isinstance(parameters[key], tuple):
            args = (key, ) + parameters[key]
            print('{:20s} : {:10.4f} +- {:5.2f}'.format(*args))
        else:
            print('{:20s} : {:10.4f}'.format(key, + parameters[key]))

    with open(os.sep.join([dirname, 'shockgeometry.json']), 'w') as fp:
        fp.write(json.dumps(parameters, indent=4))

    return result


def preprocess():
    vardict = pytplot.data_quants
    Bf = [0] * 4
    Ni = [0] * 4
    Ne = [0] * 4
    Vi = [0] * 4
    Ve = [0] * 4
    Pi = [0] * 4
    Pe = [0] * 4
    for i in range(4):
        sc = 'mms%d_' % (i+1)
        Bf[i] = vardict[sc + 'fgm_b_gse_srvy_l2']
        Ni[i] = vardict[sc + 'dis_numberdensity_fast']
        Ne[i] = vardict[sc + 'des_numberdensity_fast']
        Vi[i] = vardict[sc + 'dis_bulkv_gse_fast']
        Ve[i] = vardict[sc + 'des_bulkv_gse_fast']
        Pi[i] = vardict[sc + 'dis_prestensor_gse_fast']
        Pe[i] = vardict[sc + 'des_prestensor_gse_fast']

    #
    # (1) downsample magnetic field
    # (2) interpolate moment quantities
    # (3) average over four spacecraft
    #
    dt = 4.5 * 1.0e+9
    tt = Bf[0].time.values
    nn = (tt[-1] - tt[0])/dt
    tb = np.arange(nn) * dt + tt[0]
    tc = tb[:-1] + 0.5*(tb[+1:] - tb[:-1])

    bf = aspy.create_xarray(x=tc, y=np.zeros((tc.size, 3)))
    ni = aspy.create_xarray(x=tc, y=np.zeros((tc.size,)))
    ne = aspy.create_xarray(x=tc, y=np.zeros((tc.size,)))
    vi = aspy.create_xarray(x=tc, y=np.zeros((tc.size, 3)))
    ve = aspy.create_xarray(x=tc, y=np.zeros((tc.size, 3)))
    pi = aspy.create_xarray(x=tc, y=np.zeros((tc.size,)))
    pe = aspy.create_xarray(x=tc, y=np.zeros((tc.size,)))
    for i in range(4):
        bf += 0.25 * Bf[i].groupby_bins('time', tb).mean().values[:,0:3]
        ni += 0.25 * Ni[i].interp(time=tc).values
        ne += 0.25 * Ne[i].interp(time=tc).values
        vi += 0.25 * Vi[i].interp(time=tc).values
        ve += 0.25 * Ve[i].interp(time=tc).values
        pi += 0.25 * np.trace(Pi[i].interp(time=tc).values, axis1=1, axis2=2)
        pe += 0.25 * np.trace(Pe[i].interp(time=tc).values, axis1=1, axis2=2)

    return dict(bf=bf, ni=ni, ne=ne, vi=vi, ve=ve, pi=pi, pe=pe)


def analyze_interval(trange, analyzer):
    fmt = '%Y-%m-%d %H:%M:%S'
    t1 = (pd.to_datetime(trange[0]) - np.timedelta64(10, 'm')).strftime(fmt)
    t2 = (pd.to_datetime(trange[1]) + np.timedelta64(10, 'm')).strftime(fmt)

    dirname = trange[0].strftime(DIR_FMT) + '-' + trange[1].strftime(DIR_FMT)
    if not (os.path.exists(dirname) and os.path.isdir(dirname)):
        print('ignoreing {} as it is not a directory'.format(dirname))
        return

    ## load data
    import download
    download.load_hdf5(os.sep.join([dirname, 'fast.h5']), tplot=True)
    download.load_hdf5(os.sep.join([dirname, 'omni.h5']), tplot=True)

    ## preprocess data
    data_dict = preprocess()

    ## try to determine shock parameters
    result = analyzer(trange, data_dict, dirname)

    ## clear
    pytplot.del_data()

    return result


if __name__ == '__main__':
    # error check
    if len(sys.argv) < 2:
        print('Error: No input file was specified.')
        sys.exit(-1)

    anlyzer = MixedVMCAnalyzer(3, np.timedelta64(90, 's'))

    # analyze for each file
    for fn in sys.argv[1:]:
        import download
        if os.path.isfile(fn):
            tr1, tr2 = download.read_eventlist(fn)
            for (t1, t2) in zip(tr1, tr2):
                analyze_interval([t1, t2], anlyzer)
        else:
            print('Error: No such file : %s' % (fn))
