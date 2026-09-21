import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks

_TNG_COSMO = {
    "flat": True,
    "Om0": 0.3089,
    "Ob0": 0.0486,
    "H0": 67.74,
    "sigma8": 0.8159,
    "ns": 0.9667,
}
cosmology.setCosmology("IllustrisTNG", **_TNG_COSMO)

_R_PARAMS = {
    "a0": 0.3203,
    "a0_p": 0.6148,
    "b0": 0.2674,
    "b0_p": 0.5452,
    "b_om": 0.1134,
    "b_om_p": 0.0000,
    "b_om_p2": 0.0000,
    "b_nu": 0.2080,
    "b_nu_p": -0.2233,
    "c0": -0.9596,
    "c_om": 16.2459,
    "c_om_p": 0.0039,
    "c_om_p2": 8.9691,
    "c_om2": -9.4979,
    "c_om2_p": -0.0005,
    "c_om2_p2": 10.6132,
    "c_nu": 0.0000,
    "c_nu_p": -0.4511,
    "c_nu2": -0.0185,
    "c_nu2_p": 0.0880,
}

_GAMMA_PARAMS = {
    "a0": 1.2222,
    "a1": 0.3515,
    "b0": -0.2864,
    "b1": 0.0778,
    "b2": -0.0562,
    "b3": 0.0041,
}


def _Om_z(z):
    _Om0 = _TNG_COSMO["Om0"]
    return _Om0 * (1.0 + z) ** 3 / (_Om0 * (1.0 + z) ** 3 + (1.0 - _Om0))


def _median_Gamma_dyn(nu200m, z):
    _params = _GAMMA_PARAMS
    _A = _params["a0"] + _params["a1"] * z
    _B = _params["b0"] + _params["b1"] * z + _params["b2"] * z**2 + _params["b3"] * z**3
    return _A * nu200m + _B * nu200m**1.5


def Diemer17_Rsp_R200m_scaler(M200m, z):
    _nu200m = peaks.peakHeight(M200m, z)
    _Om = _Om_z(z)

    _Gamma = _median_Gamma_dyn(_nu200m, z)
    _p = 0.75

    _params = _R_PARAMS
    _A0 = _params["a0"] + _p * _params["a0_p"]
    _B0 = _params["b0"] + _p * _params["b0_p"]
    _B_Om = _params["b_om"] + _params["b_om_p"] * np.exp(_p * _params["b_om_p2"])
    _B_nu = _params["b_nu"] + _p * _params["b_nu_p"]
    _C0 = _params["c0"]
    _C_Om = _params["c_om"] + _params["c_om_p"] * np.exp(_p * _params["c_om_p2"])
    _C_Om2 = _params["c_om2"] + _params["c_om2_p"] * np.exp(_p * _params["c_om2_p2"])
    _C_nu = _params["c_nu"] + _p * _params["c_nu_p"]
    _C_nu2 = _params["c_nu2"] + _p * _params["c_nu2_p"]

    A = _A0
    B = (_B0 + _B_Om * _Om) * (1.0 + _B_nu * _nu200m)
    C = (_C0 + _C_Om * _Om + _C_Om2 * _Om**2) * (
        1.0 + _C_nu * _nu200m + _C_nu2 * _nu200m**2
    )

    return A + B * np.exp(-_Gamma / C)
