"""
Role of the indirect pathway of the basal ganglia in perceptual decision making.
W Wei, JE Rubin, & X-J Wang, JNS 2015.
http://dx.doi.org/10.1523/JNEUROSCI.3611-14.2015
This example also demonstrates how to use Python's cPickle module to save and load
complex data.

Adapted from (Brian, Python2) to (Brian2, Python3) by Lethe Field
"""
from __future__ import division

import pickle
import os
import cython
import numpy as np
import random  # Import before Brian floods the namespace
import argparse
import multiprocessing
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from scipy.sparse import csr_matrix
from brian2 import *
prefs.codegen.target="numpy"
# Once code is working, turn off unit-checking for speed
# import brian_no_units

# Make Brian faster
# set_global_preferences(
#     useweave=True,
#     usecodegen=True,
#     usecodegenweave=True,
#     usecodegenstateupdate=True,
#     usenewpropagate=True,
#     usecodegenthreshold=True,
#     gcc_options=['-ffast-math', '-march=native']
#     )
#=========================================================================================
# Input arguments
#=========================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--taskID', type=int, required=True, help='Task number')
parser.add_argument('--OUTDIR', required=True, help='Output directory')
parser.add_argument('--plot', action='store_true', help='Plot with existent data')
parser.add_argument('--sim', action='store_true', help='Run simulation')
parser.add_argument('--count', action='store_true', help='Count decisions')
parser.add_argument('--coh', type=float, default=0, help='Coherence level')
parser.add_argument('--coh_c', action='store_true',
                    help='Whether coherence level will change')
parser.add_argument('--coh_s', type=float, default=2.5,
                    help='Step for coherence change')
parser.add_argument('--time', type=float, default=1.1,
                    help='Simulation time')
parser.add_argument('--Iconst', type=float, default=0.38,
                    help='Constant current input to ACC')
parser.add_argument('--Iconst_c', action='store_true')
parser.add_argument('--Iconst_s', default=0.005)
parser.add_argument('--pA_ACC', default=0.15)
parser.add_argument('--pA_ACC_c', action='store_true')
parser.add_argument('--pA_ACC_s', default=0.05)
parser.add_argument('--ACC_STN_a', default=0.47)
parser.add_argument('--ACC_STN_a_c', action='store_true')
parser.add_argument('--ACC_STN_a_s', default=0.05)
parser.add_argument('--ACC_STN_n', default=0.00)
parser.add_argument('--ACC_STN_n_c', action='store_true')
parser.add_argument('--ACC_STN_n_s', default=0.01)
parser.add_argument('--Cx_pA', default=0.42)
parser.add_argument('--Cx-pA_c', action='store_true')
parser.add_argument('--Cx_pA_s', default=0.03)
parser.add_argument('--G_STN', default=0.45)
parser.add_argument('--G_STN_c', action='store_true')
parser.add_argument('--G_STN_s', default=0.05)
parser.add_argument('--STN_SNr', default=0.074)
parser.add_argument('--STN_SNr_c', action='store_true')
parser.add_argument('--STN_SNr_s', default=0.002)
parser.add_argument('--Str_SNr', default=3.6)
parser.add_argument('--Str_SNr_c', action='store_true')
parser.add_argument('--Str_SNr_s', default=0.3)
parser.add_argument('--G_SNr', default=0.09)
parser.add_argument('--G_SNr_c', action='store_true')
parser.add_argument('--G_SNr_s', default=0.005)
parser.add_argument('--nSTN', default=2.6)
parser.add_argument('--nSTN_c', action='store_true')
parser.add_argument('--nSTN_s', default=-0.2)
parser.add_argument('--nSNr', default=0.8)
parser.add_argument('--nSNr_c', action='store_true')
parser.add_argument('--nSNr_s', default=-0.03)
opt = parser.parse_args()

#=========================================================================================
# Equations
#=========================================================================================

# sAMPA, x, sNMDA, sGABA are synaptic conductances stored pre-synaptically
# S_AMPA, S_NMDA, S_GABA are synaptic conductances stored post-synaptically

equations = dict(
    # Excitatory neurons in cerebral cortex
    E=''' 
    dV/dt         = (-(V - V_L) - Isyn/gE) / tau_m_E : volt
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_E*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_E*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dsNMDA/dt     = -sNMDA/tauNMDA : 1
    dD/dt         = (1 - D)/tauD : 1
    S_AMPA : 1
    S_NMDA : 1
    S_GABA : 1
    ''',

    # Inhibitory neurons in cerebral cortex
    # Note that sAMPA and sNMDA does not decay in this group
    I='''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : volt
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_AMPA: 1
    S_NMDA: 1
    S_GABA: 1
    ''',


    Str='''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : volt
    Isyn          = I_AMPA_ext + I_AMPA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : amp
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_AMPA: 1
    S_GABA: 1
    ''',

    GPe='''
    dV/dt         = (-(V - V_L) - I_T/gI - Isyn/gI) / tau_m_I : volt
    Isyn          = I_AMPA_ext + I_GABA_ext + I_AMPA + I_NMDA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : amp
    I_GABA_ext    = gGABA_ext_I*sGABA_ext*(V - V_I) : amp
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : amp
    I_T           = gT*h*(V>V_h)*(V-V_T) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA_ext/dt = -sGABA_ext/tauGABA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    dh/dt         = -h/tauhminus*(V>=V_h) + (1-h)/tauhplus*(V<V_h) : 1
    S_AMPA: 1
    S_NMDA: 1
    S_GABA: 1
    ''',

    STN='''
    dV/dt         = (-(V - V_L) - I_T/gE - Isyn/gE) / tau_m_E : volt
    Isyn          = I_AMPA_ext + I_GABA + I_NMDA + I_AMPA : amp
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : amp
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : amp
    I_AMPA        = gAMPA_E*S_AMPA*(V-V_E) : amp
    I_NMDA        = gNMDA_E*S_NMDA*(V-V_E) : amp
    I_T           = gT*h*(V>V_h)*(V-V_T) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dsNMDA/dt     = -sNMDA/tauNMDA : 1
    dh/dt         = -h/tauhminus*(V>=V_h) + (1-h)/tauhplus*(V<V_h) :1
    S_GABA : 1
    S_NMDA : 1
    S_AMPA : 1
    ''',

    SNr='''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I :  volt
    Isyn          = I_AMPA_ext + I_NMDA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : amp
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_NMDA: 1
    S_GABA: 1
    ''',

    preA='''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : volt (unless refractory)
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA : amp
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_AMPA : 1
    S_NMDA : 1
    ''',

    ACC='''
    dV/dt         = (-(V - V_L) - Isyn/gE) / tau_m_E : volt (unless refractory)
    Isyn          = I_AMPA_ext - I_const + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : amp
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dsNMDA/dt     = -sNMDA/tauNMDA : 1
    S_GABA : 1
    ''',

    SCE='''
    dV/dt         = (-(V - V_L) - Isyn/gE) / tau_m_E : volt
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_E*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_E*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsNMDA/dt     = -sNMDA/tauNMDA : 1
    dF/dt         = -F/tauF :1
    S_AMPA : 1
    S_NMDA : 1
    S_GABA : 1
    ''',

    SCI='''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : volt
    Isyn          = I_AMPA_ext + I_NMDA: amp
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : amp
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_NMDA: 1
    '''
)

#=========================================================================================
# Parameters
#=========================================================================================

modelparams = {}

modelparams['Cx'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Excitatory LIF
    gE=25*nS,
    tau_m_E=20*ms,
    tau_ref_E=2*ms,

    # Inhibitory LIF
    gI=20*nS,
    tau_m_I=10*ms,
    tau_ref_I=1*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,

    # Synaptic time constants
    tauAMPA=2*ms,
    tauNMDA=100*ms,
    tauGABA=5*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_E=2.1*nS,  # This will be reduced to 2.0 to make the Cx recurrent connection not enough to support persistent activity
    gAMPA_ext_I=1.62*nS,

    # Unscaled recurrent synaptic conductances (onto excitatory)
    gAMPA_E=80.0*nS,
    gNMDA_E=264.0*nS,
    gGABA_E=520.0*nS,

    # Unscaled recurrent synaptic conductances (onto inhibitory)
    gAMPA_I=64*nS,
    gNMDA_I=208*nS,
    gGABA_I=400*nS,

    # Background noise
    nu_ext=2.4*kHz,

    # Number of neurons
    N_E=1600,
    N_I=400,

    # Fraction of selective neurons
    fsel=0.15,

    # Hebb-strengthened weight
    wp=1.7,

    # STD
    tauD=600*ms,

    gNMDA_SCE_CxE=0.05*nS,  # From SCE to CxE
    gNMDA_SCE_CxI=0.11*nS  # From SCE to CxE
)

modelparams['Str'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Projection LIF
    gI=25*nS,
    tau_m_I=20*ms,
    tau_ref_PJ=0*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,


    # Synaptic time constants
    tauAMPA=2*ms,
    tauGABA=5*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I=4.0*nS,
    # Background Possion rate
    nu_ext=0.8*kHz,

    # Scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I=3.0*nS,  # From Cx, this varies from 1.0 to 4.5
    gNMDA_I=0.0*nS,  # From Cx
    gGABA_I=1.0*nS,  # From within Str

    # Number of neurons
    N_PJ=250*2
)

modelparams['SNr'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Projection LIF
    gI=25*nS,
    tau_m_I=20*ms,
    tau_ref_PJ=0*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,

    # Synaptic time constants
    tauAMPA=2*ms,
    tauGABA=5*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I=14*nS,

    # Background Possion rate
    nu_ext=(opt.nSNr+(opt.taskID-1)*float(opt.nSNr_c)
            * opt.nSNr_s)*kHz,

    # scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I=0.0*nS,  # From STN
    gNMDA_I=(opt.STN_SNr+(opt.taskID-1)*float(opt.STN_SNr_c)
             * opt.STN_SNr_s)*nS,  # From STN
    gGABA_I=(opt.Str_SNr+(opt.taskID-1)*float(opt.Str_SNr_c)
             * opt.Str_SNr_s)*nS,  # From Str

    gGABA_GPe_SNr=(opt.G_SNr+(opt.taskID-1)*float(opt.G_SNr_c)
                   * opt.G_SNr_s)*nS,  # from GPe

    # Number of neurons
    N_PJ=250*2
)

modelparams['GPe'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Projection LIF
    gI=25*nS,
    tau_m_I=20*ms,
    tau_ref_PJ=0*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,

    # IFB model parameters
    V_T=120*mV,
    V_h=-60*mV,
    gT=60*nS,
    tauhminus=20*ms,
    tauhplus=100*ms,

    # Synaptic time constants
    tauAMPA=2*ms,
    tauGABA=5*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I=3.0*nS,

    # Background Possion rate
    nu_ext_AMPA=3.2*kHz,

    # External synaptic conductances
    gGABA_ext_I=2.0*nS,

    # Background Possion rate
    nu_ext_GABA=2.0*kHz,

    # scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I=0.05*nS,  # From STN
    gNMDA_I=2.0*nS,  # From STN
    gGABA_I=4.0*nS,  # From Str, this varies from 0 to 8

    gGABA_GPe_GPe=1.5*nS,

    # Number of neurons
    N_PJ=2500*2
)

modelparams['STN'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Projection LIF
    gE=25*nS,
    tau_m_E=20*ms,
    tau_ref_PJ=0*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,

    # IFB model parameters
    V_T=120*mV,
    V_h=-60*mV,
    gT=60*nS,
    tauhminus=20*ms,
    tauhplus=100*ms,

    # Synaptic time constants
    tauAMPA=2*ms,
    tauNMDA=100*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_E=1.6*nS,

    # Background Possion rate
    nu_ext=(opt.nSTN+(opt.taskID-1)*float(opt.nSTN_c)
            * opt.nSTN_s)*kHz,

    # Scaled recurrent synaptic conductances (onto projection neurons)
    gGABA_E=(opt.G_STN+(opt.taskID-1)*float(opt.G_STN_c)
             * opt.G_STN_s)*nS,  # From GPe
    gAMPA_E=(opt.ACC_STN_a+(opt.taskID-1)*float(opt.ACC_STN_a_c)
             * opt.ACC_STN_a_s)*nS,  # From ACC
    gNMDA_E=(opt.ACC_STN_n+(opt.taskID-1)*float(opt.ACC_STN_n_c)
             * opt.ACC_STN_n_s)*nS,  # From ACC

    # Number of neurons
    N_PJ=2500*2
)

modelparams['preA'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Projection LIF
    gI=25*nS,
    tau_m_I=20*ms,
    tau_ref_PJ=0*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,


    # Synaptic time constants
    tauAMPA=2*ms,
    tauGABA=5*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I=4.0*nS,
    # Background Possion rate
    nu_ext=0.8*kHz,

    # Scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I=(opt.Cx_pA+(opt.taskID-1)*float(opt.Cx_pA_c)*opt.Cx_pA_s) * \
    nS,  # From Cx, this varies from 1.0 to 4.5
    gNMDA_I=0.0*nS,  # From Cx
    gGABA_I=1.0*nS,  # From within preA

    # Number of neurons
    N_PJ=250
)

modelparams['ACC'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Projection LIF
    gE=25*nS,
    tau_m_E=20*ms,
    tau_ref_PJ=0*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,

    # Synaptic time constants
    tauAMPA=2*ms,
    tauNMDA=100*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_E=0.6*nS,

    # Background Possion rate
    nu_ext=2.28*kHz,

    # Constant bias
    I_const=(opt.Iconst+(opt.taskID-1)*float(opt.Iconst_c)*opt.Iconst_s)*namp,

    # Scaled recurrent synaptic conductances (onto projection neurons)
    gGABA_E=(opt.pA_ACC+(opt.taskID-1)*float(opt.pA_ACC_c)
             * opt.pA_ACC_s)*nS,       # From preA

    # STF parameter
    tauF=1000*ms,

    # Number of neurons
    N_PJ=250
)


modelparams['SCE'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Projection LIF
    gE=25*nS,
    tau_m_E=20*ms,
    tau_ref_PJ=0*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,

    # Synaptic time constants
    tauAMPA=2*ms,
    tauNMDA=100*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_E=0.19*nS,

    # Background Possion rate
    nu_ext=1.28*kHz,

    # Scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_E=3.3*nS,       # From Cx
    gGABA_E=2.5*nS,       # From SNr
    gNMDA_E=1.3*nS,       # From SCE to SCE
    gGABA_SCI_SCE=2.5*nS,  # From SCI to SCE

    # STF parameter
    tauF=1000*ms,

    # Number of neurons
    N_PJ=250*2
)

modelparams['SCI'] = dict(
    # Common LIF
    V_L=-70*mV,
    Vth=-50*mV,
    Vreset=-55*mV,

    # Projection LIF
    gI=25*nS,
    tau_m_I=20*ms,
    tau_ref_PJ=0*ms,

    # Reversal potentials
    V_E=0*mV,
    V_I=-70*mV,

    # NMDA nonlinearity
    a=0.062*mV**-1,
    b=3.57,

    # Synaptic time constants
    tauAMPA=2*ms,
    tauGABA=5*ms,
    delay=0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I=2.0*nS,

    # Background Possion rate
    nu_ext=1.28*kHz,

    # scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I=0.0*nS,  # From SCE
    gNMDA_I=0.7*nS,  # From SCE
    gGABA_I=0.0*nS,  # No recurrent SCI -> SCI

    # Number of neurons
    N_PJ=250
)

#=========================================================================================
# Model
#=========================================================================================

# Stimulus input to Cxe


class Stimulus(object):
    def __init__(self, Ton, Toff, mu0, muA, muB, coh):
        self.Ton = Ton
        self.Toff = Toff
        self.mu0 = mu0
        self.muA = muA
        self.muB = muB

        #set coherence
        self.set_coh(coh)

    def s1(self, T):
        t_array = np.arange(0, T + defaultclock.dt, defaultclock.dt)
        vals = np.zeros_like(t_array) * Hz
        vals[np.logical_and(self.Ton <= t_array, t_array <
                            self.Toff)] = self.pos
        return TimedArray(vals, defaultclock.dt)

    def s2(self, T):
        t_array = np.arange(0, T + defaultclock.dt, defaultclock.dt)
        vals = np.zeros_like(t_array) * Hz
        vals[np.logical_and(self.Ton <= t_array, t_array <
                            self.Toff)] = self.neg
        return TimedArray(vals, defaultclock.dt)

    def set_coh(self, coh):
        self.pos = self.mu0 + self.muA*coh/100
        self.neg = self.mu0 - self.muB*coh/100


class Model(object):
    def __init__(self, modelparams, stimulus, T):
        #---------------------------------------------------------------------------------
        # Complete the model specification
        #---------------------------------------------------------------------------------

        # Model parameters
        params = modelparams.copy()

        # Rescale conductances by number of neurons
        for x in ['gAMPA_E', 'gAMPA_I', 'gNMDA_E', 'gNMDA_I']:
            params['Cx'][x] /= params['Cx']['N_E']
        for x in ['gGABA_E', 'gGABA_I']:
            params['Cx'][x] /= params['Cx']['N_I']

        # Make local variables for convenience
        N_E = params['Cx']['N_E']
        fsel = params['Cx']['fsel']
        wp = params['Cx']['wp']
        delay = params['Cx']['delay']

        # Subpopulation size
        N1 = int(fsel*N_E)
        N2 = N1
        N0 = N_E - (N1 + N2)
        params['Cx']['N0'] = N0
        params['Cx']['N1'] = N1
        params['Cx']['N2'] = N2

        # Hebb-weakened weight
        wm = (1 - wp*fsel)/(1 - fsel)
        params['Cx']['wm'] = wm

        # Synaptic weights between populations
        self.W = np.asarray([
            [1,  1,  1],
            [wm, wp, wm],
            [wm, wm, wp]
        ])

        #---------------------------------------------------------------------------------
        # Neuron populations
        #---------------------------------------------------------------------------------

        net = OrderedDict()  # Network objects
        netPJsub = OrderedDict()  # Projection neuron subpopulations

        for x in ['E', 'I']:
            net['Cx'+x] = NeuronGroup(params['Cx']['N_'+x],
                                      equations[x],
                                      threshold='V > Vth',
                                      reset='V=Vreset',
                                      refractory=params['Cx']['tau_ref_'+x],
                                      method='rk2',
                                      namespace=params['Cx'])
        # Excitatory subpopulations
        netPJsub['Cx0'] = net['CxE'][:params['Cx']['N0']]
        netPJsub['Cx1'] = net['CxE'][params['Cx']['N0']:params['Cx']['N0'] + params['Cx']['N1']]
        netPJsub['Cx2'] = net['CxE'][params['Cx']['N0'] + params['Cx']['N1']:]

        for x in ['Str', 'SNr', 'GPe', 'STN', 'SCE', 'SCI', 'preA', 'ACC']:
            net[x] = NeuronGroup(params[x]['N_PJ'],
                                 equations[x],
                                 threshold='V > Vth',
                                 reset='V = Vreset',
                                 refractory=params[x]['tau_ref_PJ'],
                                 method='rk2',
                                 namespace=params[x])
            if x != 'SCI' and x != 'preA' and x != 'ACC':
                netPJsub[x+'1'] = net[x][: params[x]['N_PJ']//2]
                netPJsub[x+'2'] = net[x][params[x]['N_PJ']//2:]
        netPJsub['preA'] = net['preA']
        netPJsub['ACC'] = net['ACC']
        #---------------------------------------------------------------------------------
        # Background input (post-synaptic)
        #---------------------------------------------------------------------------------

        for x in ['E', 'I']:
            net['pg'+x] = PoissonGroup(params['Cx']
                                       ['N_'+x], params['Cx']['nu_ext'])
            net['ic'+x] = Synapses(net['pg'+x], net['Cx'+x],
                                   on_pre='sAMPA_ext += 1', delay=delay)
            net['ic'+x].connect(condition='i == j')

        for x in ['Str', 'SNr', 'STN', 'SCE', 'SCI', 'preA', 'ACC']:
            net['pg'+x] = PoissonGroup(params[x]['N_PJ'], params[x]['nu_ext'])
            net['ic'+x] = Synapses(net['pg'+x], net[x],
                                   on_pre='sAMPA_ext += 1', delay=delay)
            net['ic'+x].connect(condition='i == j')

        net['pg'+'GPe_AMPA'] = PoissonGroup(params['GPe']
                                            ['N_PJ'], params['GPe']['nu_ext_AMPA'])
        net['ic'+'GPe_AMPA'] = Synapses(net['pg'+'GPe_AMPA'],
                                        net['GPe'], on_pre='sAMPA_ext += 1', delay=delay)
        net['ic'+'GPe_AMPA'].connect(condition='i == j')

        net['pg'+'GPe_GABA'] = PoissonGroup(params['GPe']
                                            ['N_PJ'], params['GPe']['nu_ext_GABA'])
        net['ic'+'GPe_GABA'] = Synapses(net['pg'+'GPe_GABA'],
                                        net['GPe'], on_pre='sGABA_ext += 1', delay=delay)
        net['ic'+'GPe_GABA'].connect(condition='i == j')

        #---------------------------------------------------------------------------------
        # Recurrent input
        #---------------------------------------------------------------------------------

        # Change pre-synaptic variables
        for x in ['CxI', 'SNr', 'GPe', 'SCI', 'Str', 'preA']:
            net['icGABA_'+x] = Synapses(net[x], net[x],
                                        on_pre='sGABA += 1', delay=delay)
            net['icGABA_'+x].connect(condition='i==j')

        # CxE
        net['icAMPA_NMDA_CxE'] = Synapses(net['CxE'], net['CxE'], on_pre='''sAMPA += 1
            sNMDA += 0.63*(1-sNMDA)
            ''', delay=delay)  # D += -0.45*D
        net['icAMPA_NMDA_CxE'].connect(condition='i==j')

        # ACC
        net['icAMPA_NMDA_ACC'] = Synapses(net['ACC'], net['ACC'], on_pre='''sAMPA += 1
            sNMDA += 0.63*(1-sNMDA)
            ''', delay=delay)  # D += -0.45*D
        net['icAMPA_NMDA_ACC'].connect(condition='i==j')

        # STN
        net['icAMPA_NMDA_STN'] = Synapses(net['STN'], net['STN'], on_pre='''sAMPA += 1
            sNMDA += 0.63*(1-sNMDA)''', delay=delay)  # alpha*(1-sNMDA)
        net['icAMPA_NMDA_STN'].connect(condition='i==j')

        # SCE
        net['icNMDA_SCE'] = Synapses(net['SCE'], net['SCE'], on_pre='''sNMDA += 0.63*(1-sNMDA)
         F += 0.15*(1-F)''', delay=delay)
        net['icNMDA_SCE'].connect(condition='i==j')

        # sparse recurrent connection
        prob_GPe_GPe = 0.05
        prob_GPe_STN = 0.02  # GPe to STN
        prob_STN_GPe = 0.05  # STN to GPe

        N_PJ1 = params['GPe']['N_PJ']//2

        # here the seed 100 is choosed arbitarily
        rns = np.random.RandomState(100)

        conn_GPe_GPe = 1*(rns.random_sample((N_PJ1, N_PJ1)) < prob_GPe_GPe)
        conn_GPe_STN = 1*(rns.random_sample((N_PJ1, N_PJ1)) < prob_GPe_STN)
        conn_STN_GPe = 1*(rns.random_sample((N_PJ1, N_PJ1)) < prob_STN_GPe)

        self.sconn_GPe_GPe = csr_matrix(conn_GPe_GPe)
        self.sconn_GPe_STN = csr_matrix(conn_GPe_STN)
        self.sconn_STN_GPe = csr_matrix(conn_STN_GPe)

        self.wGABA_GPe_SNr = params['SNr']['gGABA_GPe_SNr'] / \
            params['SNr']['gGABA_I']
        self.wGABA_GPe_GPe = params['GPe']['gGABA_GPe_GPe'] / \
            params['GPe']['gGABA_I']
        self.wNMDA_SCE_CxE = params['Cx']['gNMDA_SCE_CxE'] / \
            params['Cx']['gNMDA_E']
        self.wNMDA_SCE_CxI = params['Cx']['gNMDA_SCE_CxI'] / \
            params['Cx']['gNMDA_I']
        self.wGABA_SCI_SCE = params['SCE']['gGABA_SCI_SCE'] / \
            params['SCE']['gGABA_E']

        # Link pre-synaptic variables to post-synaptic variables
        @network_operation(when='start')
        def recurrent_input():
            SAMPA = {i: {} for i in ['1', '2']}
            SNMDA = {i: {} for i in ['1', '2']}
            SGABA = {i: {} for i in ['1', '2']}

            for x in ['Cx', 'STN']:
                for i in ['1', '2']:
                    # Sum for all neurons in the group
                    SAMPA[i][x] = sum(self.netPJsub[x+i].sAMPA)
                    SNMDA[i][x] = sum(self.netPJsub[x+i].sNMDA)

            for x in ['Str', 'SNr', 'GPe']:
                for i in ['1', '2']:
                    SGABA[i][x] = sum(self.netPJsub[x+i].sGABA)

            for i in ['1', '2']:
                SNMDA[i]['SCE'] = sum(self.netPJsub['SCE'+i].sNMDA)

            SGABA['SCI'] = sum(self.net['SCI'].sGABA)
            SAMPA['ACC'] = sum(self.net['ACC'].sAMPA)
            SNMDA['ACC'] = sum(self.net['ACC'].sNMDA)
            SGABA['preA'] = sum(self.net['preA'].sGABA)

            SCx0_AMPA = sum(self.netPJsub['Cx0'].sAMPA)
            SCx0_NMDA = sum(self.netPJsub['Cx0'].sNMDA)

            # Calculating post-synaptic variables in Cx
            # AMPA
            S = self.W.dot([SCx0_AMPA, SAMPA['1']['Cx'], SAMPA['2']['Cx']])
            for i in range(3):
                self.netPJsub['Cx'+str(i)].S_AMPA = S[i]
            self.net['CxI'].S_AMPA = S[0]

            # NMDA
            S = self.W.dot([SCx0_NMDA, SNMDA['1']['Cx'], SNMDA['2']['Cx']])
            for i in range(3):
                self.netPJsub['Cx'+str(i)].S_NMDA = S[i] + \
                    self.wNMDA_SCE_CxE*(SNMDA['1']['SCE']+SNMDA['2']['SCE'])
            self.net['CxI'].S_NMDA = S[0] + self.wNMDA_SCE_CxI * \
                (SNMDA['1']['SCE']+SNMDA['2']['SCE'])

            # GABA
            S = sum(self.net['CxI'].sGABA)
            self.net['CxE'].S_GABA = S
            self.net['CxI'].S_GABA = S

            for i in ['1', '2']:
                # For SCE -> SCI
                SNMDA[i]['SCE_F'] = dot(
                    self.netPJsub['SCE'+i].F, self.netPJsub['SCE'+i].sNMDA)

                # Str
                self.netPJsub['Str'+i].S_AMPA = dot(
                    self.netPJsub['Cx'+i].D, self.netPJsub['Cx'+i].sAMPA)
                self.netPJsub['Str'+i].S_GABA = SGABA[i]['Str']

                # SNr
                self.netPJsub['SNr'+i].S_NMDA = SNMDA[i]['STN']
                self.netPJsub['SNr'+i].S_GABA = SGABA[i]['Str'] + \
                    self.wGABA_GPe_SNr*SGABA[i]['GPe']

                # GPe
                self.netPJsub['GPe'+i].S_AMPA = self.sconn_STN_GPe.dot(
                    array(self.netPJsub['STN'+i].sAMPA))
                self.netPJsub['GPe'+i].S_NMDA = self.sconn_STN_GPe.dot(
                    array(self.netPJsub['STN'+i].sNMDA))
                self.netPJsub['GPe'+i].S_GABA = self.sconn_GPe_GPe.dot(
                    array(self.netPJsub['GPe'+i].sGABA))*self.wGABA_GPe_GPe + SGABA[i]['Str']
                # STN:
                self.netPJsub['STN'+i].S_GABA = self.sconn_GPe_STN.dot(
                    array(self.netPJsub['GPe'+i].sGABA))
                self.netPJsub['STN'+i].S_AMPA = SAMPA['ACC']
                self.netPJsub['STN'+i].S_NMDA = SNMDA['ACC']

                # SC
                self.netPJsub['SCE'+i].S_AMPA = SAMPA[i]['Cx']
                self.netPJsub['SCE'+i].S_NMDA = SNMDA[i]['SCE']
                self.netPJsub['SCE'+i].S_GABA = SGABA[i]['SNr'] + \
                    self.wGABA_SCI_SCE*SGABA['SCI']

            self.net['SCI'].S_NMDA = SNMDA['1']['SCE_F']+SNMDA['2']['SCE_F']
            self.net['preA'].S_AMPA = SAMPA['1']['Cx']+SAMPA['2']['Cx']
            self.net['preA'].S_NMDA = SNMDA['1']['Cx']+SNMDA['2']['Cx']
            self.net['ACC'].S_GABA = SGABA['preA']
        #---------------------------------------------------------------------------------
        # External input (post-synaptic)
        #---------------------------------------------------------------------------------
        global s1
        s1 = stimulus.s1(T)
        global s2
        s2 = stimulus.s2(T)
        for ind, sname in zip([1, 2], ['s1', 's2']):
            net['pg'+str(ind)] = PoissonGroup(params['Cx']
                                              ['N'+str(ind)], '%s(t)' % sname)
            net['ic'+str(ind)] = Synapses(net['pg'+str(ind)],
                                          netPJsub['Cx'+str(ind)], on_pre='sAMPA_ext += 1', delay=delay)
            net['ic'+str(ind)].connect(condition='i == j')

        #---------------------------------------------------------------------------------
        # Record rates
        #---------------------------------------------------------------------------------

        rates = OrderedDict()
        for x in netPJsub:
            rates[x] = PopulationRateMonitor(
                netPJsub[x])  # bin have to be changed

        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        self.params = params
        self.net = net
        self.netPJsub = netPJsub
        self.rates = rates

        # Add network objects and monitors to NetworkOperation's contained_objects
        self.contained_objects = list()
        self.contained_objects.extend(self.net.values())
        self.contained_objects.extend(self.rates.values())
        self.contained_objects.extend([recurrent_input])

    def reinit(self):
        # Randomly initialize membrane potentials
        for x in ['E', 'I']:
            self.net['Cx'+x].V = np.random.uniform(self.params['Cx']['Vreset'],
                                                   self.params['Cx']['Vth'],
                                                   size=self.params['Cx']['N_'+x]) * volt
        for x in ['Str', 'SNr', 'GPe', 'STN', 'SCE', 'SCI', 'preA', 'ACC']:
            self.net[x].V = np.random.uniform(self.params[x]['Vreset'],
                                              self.params[x]['Vth'],
                                              size=self.params[x]['N_PJ']) * volt

        # Set synaptic variables to zero
        for i in ['CxE', 'STN', 'ACC']:
            for x in ['sAMPA_ext', 'sAMPA', 'sNMDA']:
                setattr(self.net[i], x, 0)
        for i in ['CxI', 'Str', 'SNr', 'SCI', 'preA']:
            for x in ['sAMPA_ext', 'sGABA']:
                setattr(self.net[i], x, 0)
        for x in ['sAMPA_ext', 'sNMDA']:
            setattr(self.net['SCE'], x, 0)
        for x in ['sAMPA_ext', 'sGABA_ext', 'sGABA']:
            setattr(self.net['GPe'], x, 0)
        setattr(self.net['SCE'], 'F', 0)
        setattr(self.net['GPe'], 'h', 1)
        setattr(self.net['STN'], 'h', 1)
        setattr(self.net['CxE'], 'D', 1)

#=========================================================================================
# Simulation
#=========================================================================================


class Simulation(object):
    def __init__(self, modelparams, stimparams, sim_dt, T):
        defaultclock.dt = sim_dt
        self.stimulus = Stimulus(stimparams['Ton'], stimparams['Toff'],
                                 stimparams['mu0'], stimparams['muA'],
                                 stimparams['muB'], stimparams['coh'])
        self.model = Model(modelparams, self.stimulus, T)
        self.network = Network(self.model.contained_objects)

    def run(self, T, randseed=1):
        # Initialize random number generators
        seed(randseed)

        # Initialize and run
        self.model.reinit()
        if randseed==5:
            self.network.run(T, report='text')
        else:
            self.network.run(T)

    def saverates(self, filename):
        time = self.model.rates['Cx1'].t/ms
        rates = {}
        for name in ['Cx', 'Str', 'SNr', 'GPe', 'STN', 'SCE']:
            rates[name+'1'] = self.model.rates[name +
                                               '1'].smooth_rate(width=5*ms)/Hz
            rates[name+'2'] = self.model.rates[name +
                                               '2'].smooth_rate(width=5*ms)/Hz
        rates['preA'] = self.model.rates['preA'].smooth_rate(width=5*ms)/Hz
        rates['ACC'] = self.model.rates['ACC'].smooth_rate(width=5*ms)/Hz

        with open(filename, 'wb') as f:
            pickle.dump((time, rates), f)


#/////////////////////////////////////////////////////////////////////////////////////////
stimparams = dict(
    Ton=0.5*second,  # Stimulus onset
    Toff=1.5*second,  # Stimulus offset
    mu0=30*Hz,      # Input rate
    muA=60*Hz,
    muB=20*Hz,
    coh=opt.coh+(float(opt.coh_c)*(opt.taskID-1) * \
                 opt.coh_s)         # Percent coherence
)
sim_dt = 0.05*ms
T = opt.time*second
subdir = 'data{}'.format(opt.taskID)
##############################################################################################


def run_sim(seedi):
    sim = Simulation(modelparams, stimparams, sim_dt, T)
    sim.run(T, randseed=seedi)

    dataname = 'rates{}.pkl'.format(seedi)
    datapath = os.path.join(subdir, dataname)
    sim.saverates(datapath)

def plot_rate(time, rates, filepath):
    w = 0.23
    h = 0.20
    dx = 0.08
    dy = 0.12
    x1 = 0.1
    x2 = x1+w+dx
    x3 = x2+w+dx
    y1 = 0.1
    y2 = y1+h+dy
    y3 = y2+h+dy
    # Figure setup
    fig = plt.figure()
    plots = {
        'SNr': fig.add_axes([x1, y1, w, h]),
        'STN': fig.add_axes([x2, y1, w, h]),
        'SCE': fig.add_axes([x3, y1, w, h]),
        'Cx':  fig.add_axes([x1, y2, w, h]),
        'GPe': fig.add_axes([x2, y2, w, h]),
        'Str': fig.add_axes([x3, y2, w, h]),
        'preA': fig.add_axes([x1, y3, w, h]),
        'ACC': fig.add_axes([x2, y3, w, h])
    }
    for name, plot in plots.items():
        plot.set_title(name)
    plots['SNr'].set_xlabel('Time from stimulus (ms)')
    plots['SNr'].set_ylabel('Firing rate (Hz)')

    for name, plot in plots.items():
        if name != 'preA' and name != 'ACC':
            plot.plot(time, rates[name+'1'], 'g', zorder=5)
            plot.plot(time, rates[name+'2'], 'b', zorder=5)
        else:
            plot.plot(time, rates[name], 'r', zorder=5)
        plot.set_xlim(-100, 600)
        plot.set_xticks([0, 200, 400, 600])

    plots['Cx'].set_ylim(0, 20)
    plots['STN'].set_ylim(0, 100)
    plots['SCE'].set_ylim(0, 300)
    plots['GPe'].set_ylim(0, 120)
    plots['Str'].set_ylim(0, 45)
    plots['SNr'].set_ylim(0,200)

    plt.savefig(filepath)


if __name__ == '__main__':
    os.chdir('{}'.format(opt.OUTDIR))
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    if opt.sim:
        pool = multiprocessing.Pool(16)
        seedlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        pool.map(run_sim, seedlist)
        # run_sim(1)
        #-------------------------------------------------------------------------------------
        # Plot firing rates in different areas
        #-------------------------------------------------------------------------------------
        rates_load = {}
        # Load firing rates
        for idx in range(16):
            dataname = 'rates{}.pkl'.format(idx+1)
            datapath = os.path.join(subdir, dataname)
            with open(datapath, 'rb') as f:
                time, rates_load[idx] = pickle.load(f)

        # Average across trials
        rates = {}
        for idx in range(16):
            for name in ['Cx', 'Str', 'SNr', 'GPe', 'STN', 'SCE']:
                if idx == 0:
                    rates[name+'1'] = rates_load[idx][name+'1']/16
                    rates[name+'2'] = rates_load[idx][name+'2']/16
                else:
                    rates[name+'1'] += rates_load[idx][name+'1']/16
                    rates[name+'2'] += rates_load[idx][name+'2']/16
            if idx == 0:
                rates['preA'] = rates_load[idx]['preA']/16
                rates['ACC'] = rates_load[idx]['ACC']/16
            else:
                rates['preA'] += rates_load[idx]['preA']/16
                rates['ACC'] += rates_load[idx]['ACC']/16

        # Align time to stimulus onset
        time -= stimparams['Ton']/ms
        filename = 'figure_avg.pdf'
        filepath = os.path.join(subdir, filename)
        plot_rate(time, rates, filepath)

    if opt.plot:
        for idx in range(16):
            dataname = 'rates{}.pkl'.format(idx+1)
            datapath = os.path.join(subdir, dataname)
            with open(datapath, 'rb') as f:
                time, rates = pickle.load(f)
            time -= stimparams['Ton']/ms
            filename = 'figure_{}.pdf'.format(idx+1)
            filepath = os.path.join(subdir, filename)
            plot_rate(time, rates, filepath)

    if opt.count:
        spike_count = 0 
        spike_time = []
        spike_thre = []
        for idx in range(16):   
            dataname = 'rates{}.pkl'.format(idx+1)
            datapath = os.path.join(subdir, dataname)
            with open(datapath, 'rb') as f:
                time, rates = pickle.load(f)
            time -= stimparams['Ton']/ms

            t = 0
            while time[t]<599:
                if rates['SCE1'][t]>30 or rates['SCE2'][t]>30:
                    spike_count += 1
                    spike_time.append(time[t])
                    spike_thre.append(max(rates['Cx1'][t], rates['Cx2'][t]))
                    break
                t += 1
            
        if spike_count>0:       
            time_mean = np.mean(spike_time)
            time_sigma = np.sqrt(np.var(spike_time))
            thre_mean = np.mean(spike_thre)
            thre_sigma = np.sqrt(np.var(spike_thre))
        else:
            time_mean = -1
            time_sigma = -1
            thre_mean = -1
            thre_sigma = -1
        dataname = os.path.join(subdir, 'thre.txt')
        np.savetxt(dataname, [time_mean, time_sigma, thre_mean, thre_sigma])


        
