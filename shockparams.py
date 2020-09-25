#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shock Parameter Analysis Tool

This is a tool to estimate macroscopic parameters of earth's bow shock observed
by MMS spacecraft. Parameters obtained are:

- shock normal direction
- shock angle (angle between upstream B and shock normal)
- normal Alfvenic Mach number
- shock propagation speed in spacecraft frame

The obtained shock normal direction is used to construct the LMN coordinate
system defined as follows:

- N : normal to the shock surface pointing radially outward
- M : defined by N x L
- L : parallel to the upstream transverse magnetic field direction

N-L plane is essentially the coplanarity plane and the M direction is
anti-parallel to the convection (-V x B) electric field direction.


This accepts configuration files as command-line arguments which describe time
intervals for the analysis. Multiple files are processed one-by-one. The result
of analysis is written into a directory automatically named according to the
configuration file.

"""
import os
import sys
sys.dont_write_bytecode = True

import numpy as np
import scipy as sp
from scipy import optimize
import pandas as pd

# matplotlib
import matplotlib as mpl
from matplotlib import pylab as plt

import aspy
const = aspy.const


class NormalEstimator(object):
    def __init__(self, vars1, vars2):
        # check input tuple
        nt1 = np.unique([v.shape[0] for v in vars1])
        nt2 = np.unique([v.shape[0] for v in vars2])
        if len(nt1) == 1:
            nt1 = nt1[0]
        if len(nt2) == 1:
            nt2 = nt2[0]
        self.nt1   = nt1
        self.nt2   = nt2
        self.vars1 = vars1
        self.vars2 = vars2

    def get_pair(self):
        i, j = np.mgrid[0:self.nt1,0:self.nt2]
        ii = i.ravel()
        jj = j.ravel()
        pair = list()
        for v1, v2 in zip(self.vars1, self.vars2):
            pair.append((v1[ii], v2[jj]))
        return pair

    def normal(self):
        pass


class VC(NormalEstimator):
    def normal(self):
        pair = self.get_pair()
        U1, U2 = pair[0]
        U0 = np.sqrt(np.sum((U2 - U1)**2, axis=1))
        nv = (U2 - U1)/U0[:,None]
        r, t, p = aspy.xyz2sph(nv[:,0], nv[:,1], nv[:,2])
        return t, p


class MC(NormalEstimator):
    def normal(self):
        pair = self.get_pair()
        B1, B2 = pair[0]
        nv = np.cross(np.cross(B1, B2), B1 - B2)
        r, t, p = aspy.xyz2sph(nv[:,0], nv[:,1], nv[:,2])
        return t, p


class MixedVM(NormalEstimator):
    def normal(self):
        pair = self.get_pair()
        U1, U2 = pair[0]
        B1, B2 = pair[1]
        dB = B2 - B1
        dU = U2 - U1
        nv = np.cross(np.cross(dB, dU), dB)
        r, t, p = aspy.xyz2sph(nv[:,0], nv[:,1], nv[:,2])
        return t, p


class PartialRH(NormalEstimator):
    def __init__(self, vars1, vars2, **config):
        super(PartialRH, self).__init__(vars1, vars2)
        self.method = dict()
        self.method['rho'] = config.get('method', True)
        self.method['parameter'] = None

    def normal(self, x=None):
        pair = self.get_pair()
        R1, R2 = pair[0]
        U1, U2 = pair[1]
        P1, P2 = pair[2]
        B1, B2 = pair[3]
        t, p = self.solve(R1, R2, U1, U2, B1, B2, x)
        return t, p

    def get_shock_angle(self, nx, ny, nz):
        if self.method['parameter'] == 'all_pair':
            pair = self.get_pair()
            R1, R2 = pair[0]
            U1, U2 = pair[1]
            P1, P2 = pair[2]
            B1, B2 = pair[3]
            B0 = np.sqrt(np.sum(B1**2, axis=1))
            Bn = B1[:,0]*nx + B1[:,1]*ny + B1[:,2]*nz
        else:
            R1, U1, P1, B1 = self.vars1
            B1 = B1.mean(axis=0)
            B0 = np.sqrt(np.sum(B1**2))
            Bn = B1[0] * nx + B1[1] * ny + B1[2] * nz

        Tb = np.rad2deg(np.arccos(Bn/B0))
        return Tb

    def get_mach_number(self, nx, ny, nz, vsh=0):
        if self.method['parameter'] == 'all_pair':
            pair = self.get_pair()
            R1, R2 = pair[0]
            U1, U2 = pair[1]
            P1, P2 = pair[2]
            B1, B2 = pair[3]
            B0 = np.sqrt(np.sum(B1**2, axis=1))
            Un = U1[:,0]*nx + U1[:,1]*ny + U1[:,2]*nz
        else:
            R1, U1, P1, B1 = self.vars1
            R1 = R1.mean(axis=0)
            B1 = B1.mean(axis=0)
            U1 = U1.mean(axis=0)
            B0 = np.sqrt(np.sum(B1**2))
            Un = U1[0] * nx + U1[1] * ny + U1[2] * nz

        Vn = Un - vsh
        Va = 21.8 * B0 / np.sqrt(R1)
        Ma = np.abs(Vn / Va)
        return Ma

    def get_shock_speed_efield(self, nx, ny, nz):
        if self.method['parameter'] == 'all_pair':
            pair = self.get_pair()
            R1, R2 = pair[0]
            U1, U2 = pair[1]
            P1, P2 = pair[2]
            B1, B2 = pair[3]
            # N
            nvec = np.vstack([nx, ny, nz]).T
            # L
            Bn = np.sum(B1*nvec, axis=1)
            Bt = B1 - Bn[:,None]*nvec
            lvec = Bt / np.sqrt(np.sum(Bt**2, axis=1))[:,None]
            # M
            mvec = np.cross(nvec, lvec)
            mvec = mvec / np.sqrt(np.sum(mvec**2, axis=1))[:,None]
            # Em and Bl
            E1m = np.sum(-np.cross(U1, B1)*mvec, axis=1)
            E2m = np.sum(-np.cross(U2, B2)*mvec, axis=1)
            B1l = np.sum(B1*lvec, axis=1)
            B2l = np.sum(B2*lvec, axis=1)
            Vsh = - (E2m - E1m) / (B2l - B1l)
        else:
            R1, U1, P1, B1 = self.vars1
            R2, U2, P2, B2 = self.vars2
            U1 = U1.mean(axis=0)
            B1 = B1.mean(axis=0)
            U2 = U2.mean(axis=0)
            B2 = B2.mean(axis=0)
            # N
            nvec = np.vstack([nx, ny, nz]).T
            # L
            Bn = B1[0]*nx + B1[1]*ny + B1[2]*nz
            Bt = B1[None,:] - Bn[:,None]*nvec
            lvec = Bt / np.sqrt(np.sum(Bt**2, axis=1))[:,None]
            # M
            mvec = np.cross(nvec, lvec)
            mvec = mvec / np.sqrt(np.sum(mvec**2, axis=1))[:,None]
            # Em and Bl
            E1m = np.sum(-np.cross(U1, B1)[None,:]*mvec, axis=1)
            E2m = np.sum(-np.cross(U2, B2)[None,:]*mvec, axis=1)
            B1l = np.sum(B1*lvec, axis=1)
            B2l = np.sum(B2*lvec, axis=1)
            Vsh = - (E2m - E1m) / (B2l - B1l)

        return Vsh

    def get_shock_speed_massflux_r(self, nx, ny, nz):
        if self.method['parameter'] == 'all_pair':
            pair = self.get_pair()
            R1, R2 = pair[0]
            U1, U2 = pair[1]
            P1, P2 = pair[2]
            B1, B2 = pair[3]
            Usx = (R2*U2[:,0] - R1*U1[:,0])/(R2 - R1)
            Usy = (R2*U2[:,1] - R1*U1[:,1])/(R2 - R1)
            Usz = (R2*U2[:,2] - R1*U1[:,2])/(R2 - R1)
        else:
            R1, U1, P1, B1 = self.vars1
            R2, U2, P2, B2 = self.vars2
            R1  = R1.mean(axis=0)
            U1  = U1.mean(axis=0)
            R2  = R2.mean(axis=0)
            U2  = U2.mean(axis=0)
            Usx = (R2*U2[0] - R1*U1[0])/(R2 - R1)
            Usy = (R2*U2[1] - R1*U1[1])/(R2 - R1)
            Usz = (R2*U2[2] - R1*U1[2])/(R2 - R1)

        Vsh = Usx*nx + Usy*ny + Usz*nz
        return Vsh

    def get_shock_speed_massflux_b(self, nx, ny, nz):
        if self.method['parameter'] == 'all_pair':
            pair = self.get_pair()
            R1, R2 = pair[0]
            U1, U2 = pair[1]
            P1, P2 = pair[2]
            B1, B2 = pair[3]
            B1n = B1[:,0]*nx + B1[:,1]*ny + B1[:,2]*nz
            B2n = B2[:,0]*nx + B2[:,1]*ny + B2[:,2]*nz
            B1t = np.sqrt(np.sum(B1**2, axis=1) - B1n**2)[:,None]
            B2t = np.sqrt(np.sum(B2**2, axis=1) - B2n**2)[:,None]
            Usx = (B2t*U2[:,0] - B1t*U1[:,0])/(B2t - B1t)
            Usy = (B2t*U2[:,1] - B1t*U1[:,1])/(B2t - B1t)
            Usz = (B2t*U2[:,2] - B1t*U1[:,2])/(B2t - B1t)
        else:
            R1, U1, P1, B1 = self.vars1
            R2, U2, P2, B2 = self.vars2
            U1  = U1.mean(axis=0)
            B1  = B1.mean(axis=0)
            U2  = U2.mean(axis=0)
            B2  = B2.mean(axis=0)
            B1n = B1[0]*nx + B1[1]*ny + B1[2]*nz
            B2n = B2[0]*nx + B2[1]*ny + B2[2]*nz
            B1t = np.sqrt(np.sum(B1**2) - B1n**2)[:,None]
            B2t = np.sqrt(np.sum(B2**2) - B2n**2)[:,None]
            Usx = (B2t*U2[0] - B1t*U1[0])/(B2t - B1t)
            Usy = (B2t*U2[1] - B1t*U1[1])/(B2t - B1t)
            Usz = (B2t*U2[2] - B1t*U1[2])/(B2t - B1t)

        Vsh = Usx*nx + Usy*ny + Usz*nz
        return Vsh

    def get_shock_speed_momentumflux(self, nx, ny, nz):
        if self.method['parameter'] == 'all_pair':
            pair = self.get_pair()
            R1, R2 = pair[0]
            U1, U2 = pair[1]
            P1, P2 = pair[2]
            B1, B2 = pair[3]
            U1n = U1[:,0]*nx + U1[:,1]*ny + U1[:,2]*nz
            U2n = U2[:,0]*nx + U2[:,1]*ny + U2[:,2]*nz
            B1n = B1[:,0]*nx + B1[:,1]*ny + B1[:,2]*nz
            B2n = B2[:,0]*nx + B2[:,1]*ny + B2[:,2]*nz
            B1t = np.sum(B1**2, axis=1) - B1n**2
            B2t = np.sum(B2**2, axis=1) - B2n**2
        else:
            R1, U1, P1, B1 = self.vars1
            R2, U2, P2, B2 = self.vars2
            R1  = R1.mean(axis=0)
            U1  = U1.mean(axis=0)
            P1  = P1.mean(axis=0)
            B1  = B1.mean(axis=0)
            R2  = R2.mean(axis=0)
            U2  = U2.mean(axis=0)
            P2  = P2.mean(axis=0)
            B2  = B2.mean(axis=0)
            U1n = U1[0]*nx + U1[1]*ny + U1[2]*nz
            U2n = U2[0]*nx + U2[1]*ny + U2[2]*nz
            B1n = B1[0]*nx + B1[1]*ny + B1[2]*nz
            B2n = B2[0]*nx + B2[1]*ny + B2[2]*nz
            B1t = np.sum(B1**2) - B1n**2
            B2t = np.sum(B2**2) - B2n**2

        P1n = P1
        P2n = P2
        Ro  = (R2 - R1)*(const.mp + const.me) * 1.0e+6
        Ke  = (R2*U2n**2 - R1*U1n**2)*(const.mp + const.me) * 1.0e+12
        Pg  = (P2n - P1n) * 1.0e-9
        Pb  = (B2t - B1t)/(2*const.mu0) * 1.0e-18
        Pt  = (Ke + Pg + Pb)
        Vsh = np.sqrt(np.abs(Pt / Ro)) * np.sign(Pt) * 1.0e-3
        return Vsh

    def get_parameters(self, t, p):
        result = dict()

        # normal vector
        nx, ny, nz = aspy.sph2xyz(1, t, p)
        Nx = np.atleast_1d(nx.mean())
        Ny = np.atleast_1d(ny.mean())
        Nz = np.atleast_1d(nz.mean())
        result['theta'] = t
        result['phi'] = p

        # shock speed
        Vsh_efd = self.get_shock_speed_efield(nx, ny, nz)
        Vsh_rho = self.get_shock_speed_massflux_r(nx, ny, nz)
        Vsh_bfd = self.get_shock_speed_massflux_b(nx, ny, nz)
        Vsh_mom = self.get_shock_speed_momentumflux(nx, ny, nz)
        result['Vsh_efd'] = Vsh_efd
        result['Vsh_rho'] = Vsh_rho
        result['Vsh_bfd'] = Vsh_bfd
        result['Vsh_mom'] = Vsh_mom

        # mean shock speed
        Vsh_efd_mean = self.get_shock_speed_efield(Nx, Ny, Nz)
        Vsh_rho_mean = self.get_shock_speed_massflux_r(Nx, Ny, Nz)
        Vsh_bfd_mean = self.get_shock_speed_massflux_b(Nx, Ny, Nz)
        Vsh_mom_mean = self.get_shock_speed_momentumflux(Nx, Ny, Nz)
        result['Vsh_efd_mean'] = Vsh_efd_mean[0]
        result['Vsh_rho_mean'] = Vsh_rho_mean[0]
        result['Vsh_bfd_mean'] = Vsh_bfd_mean[0]
        result['Vsh_mom_mean'] = Vsh_mom_mean[0]

        # shock angle
        Tbn = self.get_shock_angle(nx, ny, nz)
        result['Tbn'] = Tbn

        # mean shock angle
        Tbn_mean = self.get_shock_angle(Nx, Ny, Nz)
        result['Tbn_mean'] = Tbn_mean[0]

        # Alfven Mach number
        Man_sc  = self.get_mach_number(nx, ny, nz)
        Man_efd = self.get_mach_number(nx, ny, nz, Vsh_efd)
        Man_rho = self.get_mach_number(nx, ny, nz, Vsh_rho)
        Man_bfd = self.get_mach_number(nx, ny, nz, Vsh_bfd)
        Man_mom = self.get_mach_number(nx, ny, nz, Vsh_mom)
        result['Man_sc']  = Man_sc
        result['Man_efd'] = Man_efd
        result['Man_rho'] = Man_rho
        result['Man_bfd'] = Man_bfd
        result['Man_mom'] = Man_mom

        # mean Alfven Mach number
        Man_sc_mean  = self.get_mach_number(nx, ny, nz)
        Man_efd_mean = self.get_mach_number(nx, ny, nz, Vsh_efd)
        Man_rho_mean = self.get_mach_number(nx, ny, nz, Vsh_rho)
        Man_bfd_mean = self.get_mach_number(nx, ny, nz, Vsh_bfd)
        Man_mom_mean = self.get_mach_number(nx, ny, nz, Vsh_mom)
        result['Man_sc_mean']  = Man_sc_mean[0]
        result['Man_efd_mean'] = Man_efd_mean[0]
        result['Man_rho_mean'] = Man_rho_mean[0]
        result['Man_bfd_mean'] = Man_bfd_mean[0]
        result['Man_mom_mean'] = Man_mom_mean[0]

        # upstream B-field
        result['R1'] = self.vars1[0]
        result['U1'] = self.vars1[1]
        result['P1'] = self.vars1[2]
        result['B1'] = self.vars1[3]
        result['R2'] = self.vars2[0]
        result['U2'] = self.vars2[1]
        result['P2'] = self.vars2[2]
        result['B2'] = self.vars2[3]

        return result

    def solve(self, R1, R2, U1, U2, B1, B2, x0=None):
        if R1.size == R2.size:
            nt = R1.size
        else:
            raise ValueError('Invalid input')
        if x0 is None:
            x0 = np.array([90.0, 0.0])
        R0 = 1.0e-6 # 1/m^6
        U0 = 1.0e-3 # m/s
        B0 = 1.0e+9 # T
        t = np.zeros((nt,), np.float64)
        p = np.zeros((nt,), np.float64)
        for i in range(nt):
            args = (R1[i]/R0, R2[i]/R0,
                    U1[i]/U0, U2[i]/U0,
                    B1[i]/B0, B2[i]/B0)
            res  = optimize.least_squares(self.f, x0, args=args)
            if res.status > 0:
                # make sure normal vector is radially outward
                nvec = aspy.sph2xyz(1, res.x[0], res.x[1])
                nsig = np.sign(nvec[0] + nvec[1] + nvec[2])
                nvec = np.array(nvec) * nsig
                _, t[i], p[i] = aspy.xyz2sph(*nvec)
            else:
                print('No : ', res.x)
                t[i] = None
                p[i] = None
        return t, p

    def f(self, x, *args):
        R1, R2, U1, U2, B1, B2 = args
        Nv = np.array(aspy.sph2xyz(1, x[0], x[1]))
        if self.method['rho']:
            # shock speed estimate using density
            Vs = (R2*U2 - R1*U1)/(R2 - R1)
        else:
            # shock speed estimate using magnetic field
            B1n = np.dot(B1, Nv)
            B2n = np.dot(B2, Nv)
            B1t = np.sqrt(np.sum(B1**2) - B1n**2)
            B2t = np.sqrt(np.sum(B2**2) - B1n**2)
            Vs = (B2t*U2 - B1t*U1)/(B2t - B1t)
        rh1 = self.partial_rh(R1, U1, B1, Vs, Nv)
        rh2 = self.partial_rh(R2, U2, B2, Vs, Nv)
        return rh1 - rh2

    def partial_rh(self, R, U, B, V, nv):
        Bn = np.dot(B, nv)
        Bt = B - Bn*nv
        Un = np.dot(U, nv)
        Ut = U - Un*nv
        Vn = np.dot(V, nv)
        St = R*(const.mp + const.me)*(Un - Vn)*Ut - Bn*Bt/const.mu0
        Et = Bn*np.cross(nv, Ut) - (Un - Vn)*np.cross(nv, Bt)
        # convert to appropriate unit
        Bn = Bn * 1.0e+9 # nT
        St = St * 1.0e+9 # nPa
        Et = Et * 1.0e+3 # mV/m
        return np.array([Bn, St[0], St[1], St[2], Et[0], Et[1], Et[2]])


def get_data(tr):
    import pytplot
    import pyspedas

    probe = [1, 2, 3, 4]

    # S/C position
    pyspedas.mms.mec(probe=probe, trange=tr, time_clip=True,
                     varformat='*mec_r_gse$')

    # FGM
    pyspedas.mms.fgm(probe=probe, trange=tr, data_rate='srvy', time_clip=True,
                     varformat='*fgm_b_gse*')

    # moments
    pyspedas.mms.fpi(probe=probe, trange=tr, data_rate='fast', time_clip=True,
                     datatype=['des-moms', 'dis-moms'],
                     varformat='*(numberdensity|bulkv_gse|prestensor_gse)*')

    # store data
    vardict = pytplot.data_quants
    pos = [0] * 4
    bf  = [0] * 4
    ni  = [0] * 4
    ne  = [0] * 4
    vi  = [0] * 4
    ve  = [0] * 4
    pi  = [0] * 4
    pe  = [0] * 4
    for i in range(4):
        sc = 'mms%d_' % (i+1)
        pos[i] = vardict[sc + 'mec_r_gse']
        bf[i]  = vardict[sc + 'fgm_b_gse_srvy_l2']
        ni[i]  = vardict[sc + 'dis_numberdensity_fast']
        ne[i]  = vardict[sc + 'des_numberdensity_fast']
        vi[i]  = vardict[sc + 'dis_bulkv_gse_fast']
        ve[i]  = vardict[sc + 'des_bulkv_gse_fast']
        pi[i]  = vardict[sc + 'dis_prestensor_gse_fast']
        pe[i]  = vardict[sc + 'des_prestensor_gse_fast']

    # clear
    pytplot.del_data()

    return dict(pos=pos, bf=bf, ni=ni, ne=ne, vi=vi, ve=ve, pi=pi, pe=pe)


def get_title(tu, td):
    tsu = [pd.Timestamp(t) for t in tu]
    tsd = [pd.Timestamp(t) for t in td]
    day = tsu[0].strftime('%Y-%m-%d')
    u1  = tsu[0].strftime('%H:%M:%S')
    u2  = tsu[1].strftime('%H:%M:%S')
    d1  = tsd[0].strftime('%H:%M:%S')
    d2  = tsd[1].strftime('%H:%M:%S')
    title = '%s : upstream = [%s, %s]; downstream = [%s, %s]' % \
                             (day, u1, u2, d1, d2)
    return title


def get_time_bins(Bf, dt):
    # down sampling the magnetic field
    dt  = 4.5
    t   = Bf.time.values
    n   = (t[-1] - t[0])/dt
    tb  = np.arange(n) * dt + t[0]
    return tb


def get_sliced_quantities(isc, tb1, tb2, args):
    Bf = args['bf'][isc]
    Ni = args['ni'][isc]
    Ne = args['ne'][isc]
    Vi = args['vi'][isc]
    Ve = args['ve'][isc]
    Pi = args['pi'][isc]
    Pe = args['pe'][isc]

    t1 = 0.5*(tb1[+1:] + tb1[:-1])
    t2 = 0.5*(tb2[+1:] + tb2[:-1])
    Pi = (Pi[:,0,0] + Pi[:,1,1] + Pi[:,2,2])/3
    Pe = (Pe[:,0,0] + Pe[:,1,1] + Pe[:,2,2])/3

    #
    # upstream
    #
    R1i = Ni.interp(time=t1).values
    U1i = Vi.interp(time=t1).values
    P1i = Pi.interp(time=t1).values
    R1e = Ne.interp(time=t1).values
    U1e = Ve.interp(time=t1).values
    P1e = Pe.interp(time=t1).values
    R1  = (R1i + R1e)/2
    U1  = U1i
    P1  = P1i + P1e
    B1  = Bf.groupby_bins('time', tb1).mean().values[:,0:3]

    u_params = (R1, U1, P1, B1)

    #
    # downstream
    #
    R2i = Ni.interp(time=t2).values
    U2i = Vi.interp(time=t2).values
    P2i = Pi.interp(time=t2).values
    R2e = Ne.interp(time=t2).values
    U2e = Ve.interp(time=t2).values
    P2e = Pe.interp(time=t2).values
    R2  = (R2i + R2e)/2
    U2  = U2i
    P2  = P2i + P2e
    B2  = Bf.groupby_bins('time', tb2).mean().values[:,0:3]

    d_params = (R2, U2, P2, B2)

    return u_params, d_params


def estimate_normal(args, sc=None, **config):
    Nm = 4
    tr1 = [pd.Timestamp(config[t]).timestamp() for t in ('tu1', 'tu2')]
    tr2 = [pd.Timestamp(config[t]).timestamp() for t in ('td1', 'td2')]

    # generate upstream-downstream pairs
    if type(sc) == int and sc >= 1 and sc <= 4:
        # use single spacecraft
        tb  = get_time_bins(args['bf'][sc-1], 4.5)
        j1  = np.searchsorted(tb, tr1)
        j2  = np.searchsorted(tb, tr2)
        tb1 = tb[j1[0]-1:j1[1]+1]
        tb2 = tb[j2[0]-1:j2[1]+1]
        u_params, d_params = get_sliced_quantities(sc-1, tb1, tb2, args)
        R1, U1, P1, B1 = u_params
        R2, U2, P2, B2 = d_params
    else:
        # average over four spacecraft
        tb  = get_time_bins(args['bf'][0], 4.5)
        j1  = np.searchsorted(tb, tr1)
        j2  = np.searchsorted(tb, tr2)
        tb1 = tb[j1[0]-1:j1[1]+1]
        tb2 = tb[j2[0]-1:j2[1]+1]
        R1 = [0]*4
        U1 = [0]*4
        P1 = [0]*4
        B1 = [0]*4
        R2 = [0]*4
        U2 = [0]*4
        P2 = [0]*4
        B2 = [0]*4
        for isc in range(4):
            u_params, d_params = get_sliced_quantities(isc, tb1, tb2, args)
            R1[isc], U1[isc], P1[isc], B1[isc] = u_params
            R2[isc], U2[isc], P2[isc], B2[isc] = d_params
        R1 = 0.25*(R1[0] + R1[1] + R1[2] + R1[3])
        U1 = 0.25*(U1[0] + U1[1] + U1[2] + U1[3])
        P1 = 0.25*(P1[0] + P1[1] + P1[2] + P1[3])
        B1 = 0.25*(B1[0] + B1[1] + B1[2] + B1[3])
        R2 = 0.25*(R2[0] + R2[1] + R2[2] + R2[3])
        U2 = 0.25*(U2[0] + U2[1] + U2[2] + U2[3])
        P2 = 0.25*(P2[0] + P2[1] + P2[2] + P2[3])
        B2 = 0.25*(B2[0] + B2[1] + B2[2] + B2[3])

    # estimate normal vector
    method    = [0]*Nm
    estimator = [0]*Nm

    method[0]    = 'Partial RH'
    estimator[0] = PartialRH((R1, U1, P1, B1), (R2, U2, P2, B2), **config)
    partial_rh   = estimator[0]

    method[1]    = 'VM Mixed'
    estimator[1] = MixedVM((U1, B1), (U2, B2))

    method[2]    = 'VC'
    estimator[2] = VC((U1,), (U2,))

    method[3]    = 'MC'
    estimator[3] = MC((B1,), (B2,))

    # estimate normal except for PartialRH
    normal = [0]*Nm
    for im in range(1, Nm):
        theta, phi = estimator[im].normal()
        normal[im] = dict(method=method[im], theta=theta, phi=phi)

    # MixedVM result is used as the initial guess for PartialRH
    im = 0
    jm = 1
    t = normal[jm]['theta'].mean()
    p = normal[jm]['phi'].mean()
    theta, phi = estimator[im].normal([t, p])
    normal[im] = dict(method=method[im], theta=theta, phi=phi)

    # now estimate shock parameters with PartialRH
    t = normal[im]['theta']
    p = normal[im]['phi']
    params = estimator[im].get_parameters(t, p)

    return normal, params


def plot_summary(isc, args, title, fn, **config):
    tr = [config['t1'], config['t2']]
    Bf = args['bf'][isc]
    Ni = args['ni'][isc]
    Ne = args['ne'][isc]
    Vi = args['vi'][isc]
    Ve = args['ve'][isc]

    # down sampling the magnetic field
    dt  = 4.5
    t   = Bf.time.values
    n   = (t[-1] - t[0])/dt
    tb  = np.arange(n) * dt + t[0]
    tt  = 0.5*(tb[+1:] + tb[:-1])
    bb  = Bf.groupby_bins('time', tb).mean().values
    bf  = aspy.create_xarray(x=tt, y=bb)

    # interpolation
    ni  = Ni.interp(time=tt)
    ne  = Ne.interp(time=tt)
    vi  = Vi.interp(time=tt)
    ve  = Ve.interp(time=tt)

    # electric field in mV/m
    Ef = np.zeros((tt.size, 3), dtype=np.float64)
    Ef[:,0] = (bf[:,1] * vi[:,2] - bf[:,2] * vi[:,1]) * 1.0e-3
    Ef[:,1] = (bf[:,2] * vi[:,0] - bf[:,0] * vi[:,2]) * 1.0e-3
    Ef[:,2] = (bf[:,0] * vi[:,1] - bf[:,1] * vi[:,0]) * 1.0e-3
    ef = aspy.create_xarray(x=tt, y=Ef)

    # set plot options
    aspy.set_plot_option(bf,
                         ylabel='B [nT]',
                         legend=('Bx', 'By', 'Bz', 'Bt'),
                         line_color=('b', 'g', 'r', 'k'),
                         trange=tr)
    aspy.set_plot_option(ef,
                         ylabel='E [mV/m]',
                         legend=('Ex', 'Ey', 'Ez'),
                         line_color=('b', 'g', 'r'),
                         trange=tr)
    aspy.set_plot_option(ni,
                         ylabel='N [1/cm^3]',
                         legend=('Ni',),
                         line_color=('r',),
                         trange=tr)
    aspy.set_plot_option(ne,
                         ylabel='N [1/cm^3]',
                         legend=('Ne',),
                         line_color=('b',),
                         trange=tr)
    aspy.set_plot_option(vi,
                         ylabel='Vi [km/s]',
                         legend=('Vix', 'Viy', 'Viz'),
                         line_color=('b', 'g', 'r'),
                         trange=tr)
    aspy.set_plot_option(ve,
                         ylabel='Ve [km/s]',
                         legend=('Vex', 'Vey', 'Vez'),
                         line_color=('b', 'g', 'r'),
                         trange=tr)

    fig = aspy.tplot([bf, [ni, ne], vi, ve, ef], title=title, backend='mpl')
    fig.savefig(fn)


def plot_normal(result, title, fn):
    Nm = 4

    method = [r['method'] for r in result]
    theta  = [r['theta'] for r in result]
    phi    = [r['phi'] for r in result]

    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.2,
                        left=0.10, right=0.95, bottom=0.05, top=0.90)
    plt.figtext(0.05, 0.95, title, horizontalalignment='left')
    ylabel = 'Occurance Rate'

    axs[Nm-1,0].set_xlabel(r'$\theta$')
    axs[Nm-1,1].set_xlabel(r'$\phi$')

    nbins = (36, 72)
    rng   = ((0.0, 180.0), (-180.0, +180.0))
    for im in range(Nm):
        t_avg  = theta[im].mean()
        t_std  = theta[im].std()
        p_avg  = phi[im].mean()
        p_std  = phi[im].std()

        # calculate and plot histogram
        thist, tbins = np.histogram(theta[im], nbins[0], range=rng[0])
        phist, pbins = np.histogram(phi[im], nbins[1], range=rng[1])
        thist = thist / np.sum(thist)
        phist = phist / np.sum(phist)

        # theta
        width = tbins[+1:] - tbins[:-1]
        trans = axs[im,0].transAxes
        label = r'$\theta$ = %6.2f +- %6.2f' % (t_avg, t_std)
        axs[im,0].set_axisbelow(True)
        axs[im,0].set_title(method[im])
        axs[im,0].set_ylabel(ylabel)
        axs[im,0].bar(tbins[:-1], thist, width=width, color='m')
        axs[im,0].text(0.05, 0.85, label, transform=trans)
        axs[im,0].set_xlim(rng[0])

        # phi
        width = pbins[+1:] - pbins[:-1]
        trans = axs[im,1].transAxes
        label = r'$\phi$ = %6.2f +- %6.2f' % (p_avg, p_std)
        axs[im,1].set_axisbelow(True)
        axs[im,1].set_title(method[im])
        axs[im,1].set_ylabel(ylabel)
        axs[im,1].bar(pbins[:-1], phist, width=width, color='m')
        axs[im,1].text(0.05, 0.85, label, transform=trans)
        axs[im,1].set_xlim(rng[1])

    fig.savefig(fn)


def plot_params(params, title, fn):
    theta = params['theta']
    phi   = params['phi']
    Vsh   = params['Vsh_efd']
    Tbn   = params['Tbn']
    Man   = params['Man_efd']

    ylabel = 'Occurance Rate'
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.10, top=0.90)
    plt.figtext(0.05, 0.95, title, horizontalalignment='left')

    # shock speed
    avg = Vsh.mean()
    std = Vsh.std()
    hist, bins = np.histogram(Vsh, 40, range=(-2.0e+2, +2.0e+2))
    hist = hist / np.sum(hist)

    width = bins[+1:] - bins[:-1]
    trans = axs[0].transAxes
    label = r'%6.2f +- %6.2f' % (avg, std)
    axs[0].set_axisbelow(True)
    axs[0].set_xlabel('shock speed [km/s]')
    axs[0].set_ylabel(ylabel)
    axs[0].bar(bins[:-1], hist, width=width, color='m', align='edge')
    axs[0].text(0.05, 0.85, label, transform=trans)
    axs[0].set_xlim(-2.0e+2, +2.0e+2)

    # shock angle
    avg = Tbn.mean()
    std = Tbn.std()
    hist, bins = np.histogram(Tbn, 90, range=(0.0, +180.0))
    hist = hist / np.sum(hist)

    width = bins[+1:] - bins[:-1]
    trans = axs[1].transAxes
    label = r'%6.2f +- %6.2f' % (avg, std)
    axs[1].set_axisbelow(True)
    axs[1].set_xlabel('shock angle [deg]')
    axs[1].set_ylabel(ylabel)
    axs[1].bar(bins[:-1], hist, width=width, color='m', align='edge')
    axs[1].text(0.05, 0.85, label, transform=trans)
    axs[1].set_xlim(0.0, 180.0)

    # Alfven Mach number
    avg = Man.mean()
    std = Man.std()
    hist, bins = np.histogram(Man, 40, range=(0.0, +20.0))
    hist = hist / np.sum(hist)

    width = bins[+1:] - bins[:-1]
    trans = axs[2].transAxes
    label = r'%6.2f +- %6.2f' % (avg, std)
    axs[2].set_axisbelow(True)
    axs[2].set_xlabel('Alfven Mach number')
    axs[2].set_ylabel(ylabel)
    axs[2].bar(bins[:-1], hist, width=width, color='m', align='edge')
    axs[2].text(0.05, 0.85, label, transform=trans)
    axs[2].set_xlim(0.0, 20.0)

    fig.savefig(fn)


def plot_position(data, tr, params, fn):
    Re    = const.Re
    pos   = data['pos']
    theta = params['theta'].mean()
    phi   = params['phi'].mean()
    B1    = params['B1'].mean(axis=0)
    b     = B1 / np.sqrt(np.sum(B1**2))
    bx, by, bz = b[0], b[1], b[2]
    nx, ny, nz = aspy.sph2xyz(1.0, theta, phi)

    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    plt.subplots_adjust(hspace=0.30, wspace=0.35,
                        left=0.06, right=0.90, bottom=0.10, top=0.90)

    ts1   = pd.Timestamp(tr[0]).strftime('%Y-%m-%d %H:%M:%S')
    ts2   = pd.Timestamp(tr[1]).strftime('%H:%M:%S')
    title = '%s - %s' % (ts1, ts2)
    plt.figtext(0.5, 0.95, title, horizontalalignment='center')

    axs = axs.ravel()
    for ax in axs:
        ax.set_aspect('equal')

    #
    # average spacecraft position
    #
    X, Y, Z = 0.0, 0.0, 0.0
    for i in range(4):
        X += pos[i].values[:,0].mean() / 4
        Y += pos[i].values[:,1].mean() / 4
        Z += pos[i].values[:,2].mean() / 4

    axs[0].plot(X/Re, Y/Re, 'o')
    axs[0].set_xlabel('GSE X [Re]')
    axs[0].set_ylabel('GSE Y [Re]')

    axs[1].plot(X/Re, Z/Re, 'o')
    axs[1].set_xlabel('GSE X [Re]')
    axs[1].set_ylabel('GSE Z [Re]')

    # show normal and b-field direction
    kwargs = dict(scale=0.2, angles='xy', scale_units='xy', color='b')
    axs[0].quiver(X/Re, Y/Re, nx, ny, **kwargs)
    axs[1].quiver(X/Re, Z/Re, nx, nz, **kwargs)

    kwargs = dict(scale=0.2, angles='xy', scale_units='xy', color='m')
    axs[0].quiver(X/Re, Y/Re, bx, by, **kwargs)
    axs[1].quiver(X/Re, Z/Re, bx, bz, **kwargs)

    # show surface perpendicular to normal
    x = np.linspace(-10.0*Re, 20.0*Re, 11)
    ny = ny + 1.0e-32
    nz = nz + 1.0e-32
    axs[0].plot(x, -nx/ny*(x-X/Re)+Y/Re, 'k--')
    axs[1].plot(x, -nx/nz*(x-X/Re)+Z/Re, 'k--')

    for ax in axs[0:2]:
        ax.set_xlim(-10.0, +20.0)
        ax.set_ylim(-15.0, +15.0)

    #
    # spacecraft separation
    #
    markers = ['ro', 'go', 'bo', 'ko']
    rmax = 0.0
    for i in range(4):
        x = pos[i].values[:,0].mean() - X
        y = pos[i].values[:,1].mean() - Y
        z = pos[i].values[:,2].mean() - Z
        rmax = np.abs(np.array([rmax, x, y, z])).max()
        axs[2].plot(x, y, markers[i])
        axs[3].plot(x, y, markers[i])

    label = ('MMS1', 'MMS2', 'MMS3', 'MMS4')
    axs[2].legend(label, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    axs[3].legend(label, loc='upper left', bbox_to_anchor=(1.0, 1.0))

    rmax = (int(rmax/5.0) + 1) * 5.0
    for ax in axs[2:4]:
        ax.set_xlim(-rmax, +rmax)
        ax.set_ylim(-rmax, +rmax)

    # show normal and b-field direction
    kwargs = dict(scale=3/rmax, angles='xy', scale_units='xy', color='b')
    axs[2].quiver(0.0, 0.0, nx, ny, **kwargs)
    axs[3].quiver(0.0, 0.0, nx, nz, **kwargs)

    kwargs = dict(scale=3/rmax, angles='xy', scale_units='xy', color='m')
    axs[2].quiver(0.0, 0.0, bx, by, **kwargs)
    axs[3].quiver(0.0, 0.0, bx, bz, **kwargs)

    # show surface perpendicular to normal
    x = np.linspace(-rmax, rmax, 11)
    ny = ny + 1.0e-32
    nz = nz + 1.0e-32
    axs[2].plot(x, -nx/ny*x, 'k--')
    axs[3].plot(x, -nx/nz*x, 'k--')

    axs[2].set_xlabel('X [km]')
    axs[2].set_ylabel('Y [km]')
    axs[3].set_xlabel('X [km]')
    axs[3].set_ylabel('Z [km]')

    plt.savefig(fn)


def get_nml_coordinate(params):
    theta = params['theta'].mean()
    phi   = params['phi'].mean()
    bup   = params['B1'].mean(axis=0)

    # N direction
    nx, ny, nz = aspy.sph2xyz(1, theta, phi)
    nvec = np.array([nx, ny, nz])

    # L direction
    bn = bup[0]*nx + bup[1]*ny + bup[2]*nz
    bt = bup - bn*nvec
    lvec = bt / np.sqrt(np.sum(bt**2))

    # M direction
    mvec = np.cross(nvec, lvec)
    mvec = mvec / np.sqrt(np.sum(mvec**2))

    # transform vector quantities to LMN coordiante
    M  = np.vstack([lvec, mvec, nvec])[None,:,:]
    U1 = params['U1'][:,None,:]
    B1 = params['B1'][:,None,:]
    U2 = params['U2'][:,None,:]
    B2 = params['B2'][:,None,:]
    U1_nml = np.sum(M*U1, axis=-1)
    B1_nml = np.sum(M*B1, axis=-1)
    U2_nml = np.sum(M*U2, axis=-1)
    B2_nml = np.sum(M*B2, axis=-1)

    params['U1_nml'] = U1_nml
    params['B1_nml'] = B1_nml
    params['U2_nml'] = U2_nml
    params['B2_nml'] = B2_nml

    return lvec, mvec, nvec


def save(fn, config, params):
    import h5py
    import json

    with h5py.File(fn, 'w') as fp:
        # config as attribute
        fp.attrs['config'] = json.dumps(config)
        # data
        for key, item in params.items():
            fp.create_dataset(key, data=item)


def printlog(log, s):
    print(s)
    return log + '# ' + s + '\n'


def process_file(cfg):
    import re
    import pickle
    import hashlib
    import configparser
    config = configparser.ConfigParser()
    config.read(cfg)

    # read configuration file
    get_config = lambda s: config.get('shockparams', s)

    config = {
        't1'  : get_config('t1'),
        't2'  : get_config('t2'),
        'tu1' : get_config('tu1'),
        'tu2' : get_config('tu2'),
        'td1' : get_config('td1'),
        'td2' : get_config('td2'),
        'method' : {
            'rho' : bool(get_config('rho')),
        },
    }

    tr  = [config['t1'], config['t2']]
    tru = [config['tu1'], config['tu2']]
    trd = [config['td1'], config['td2']]
    title   = get_title(tru, trd)
    prefix  = re.match('(.+)\.(.+)', fn).groups()[0]
    hashstr = hashlib.sha256(pickle.dumps(config)).hexdigest()[0:16]
    dirname = prefix + '_' + hashstr + '/'

    # make directory
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # read data
    data = get_data(tr)

    # summary plot
    plot_summary(0, data, title, dirname + 'summary.png', **config)

    # estimate shock normal and other parameters
    normal, params = estimate_normal(data, **config)
    plot_normal(normal, title, dirname + 'normal.png')
    plot_params(params, title, dirname + 'params.png')

    # spacecraft position
    plot_position(data, tr, params, dirname + 'position.png')

    logstr = ''
    logstr = printlog(logstr, '')
    logstr = printlog(logstr, '*** Shock Parameters Analyzed ***')
    logstr = printlog(logstr, '*** upstream interval    : [%s, %s]' % (tru[0], tru[1]))
    logstr = printlog(logstr, '*** downstream interval  : [%s, %s]' % (trd[0], trd[1]))
    logstr = printlog(logstr, '*** output directory     : %s' % (dirname))

    # mean parameters
    Tbn  = params['Tbn_mean']
    Vsh1 = params['Vsh_efd_mean']
    Vsh2 = params['Vsh_rho_mean']
    Vsh3 = params['Vsh_mom_mean']
    Man0 = params['Man_sc_mean']
    Man1 = params['Man_efd_mean']
    Man2 = params['Man_rho_mean']
    Man3 = params['Man_mom_mean']
    mean_params = {
        'theta_bn'         : Tbn,
        'V_sh (efield)'    : Vsh1,
        'V_sh (rho)'       : Vsh2,
        'V_sh (mom)'       : Vsh3,
        'M_an (sc)'        : Man0,
        'M_an (efield)'    : Man1,
        'M_an (rho)'       : Man2,
        'M_an (mom)'       : Man3,
    }
    for key, item in mean_params.items():
        logstr = printlog(logstr, '*** %-20s : %7.3f' % (key, item))

    # get LMN coordinate
    lvec, mvec, nvec = get_nml_coordinate(params)
    params['lvec'] = lvec
    params['mvec'] = mvec
    params['nvec'] = nvec

    label = ('lvec', 'mvec', 'nvec')
    for l, v in zip(label, (lvec, mvec, nvec)):
        logstr = printlog(logstr, '*** %-20s : [%+7.3f, %+7.3f, %+7.3f]'
                          % (l, v[0], v[1], v[2]))
    logstr = printlog(logstr, '')

    # write log
    with open(dirname + 'log.txt', 'w') as f:
        f.write(logstr)

    # save results
    save(dirname + 'params.h5', config, params)

    return data


if __name__ == '__main__':
    # error check
    if len(sys.argv) < 2:
        print('Error: No input file was specified.')
        sys.exit(-1)

    # analyze for each file
    for fn in sys.argv[1:]:
        if os.path.isfile(fn):
            process_file(fn)
        else:
            print('Error: No such file : %s' % (fn))
