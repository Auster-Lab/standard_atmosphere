from dataclasses import dataclass
from math import exp

AVOGADRO_CONSTANT_1_KMOL = 6.022169e26
BOLTZMANN_CONSTANT_NM_K = 1.380622e-23
IDEAL_GAS_CONSTANT_NM_KMOLK = 8.31432e3
IDEAL_GAS_GAMMA_RATIO = 1.4
SUTHERLAND_CONSTANT_K = 110
EARTH_GRAVITY_ACCELERATION_M_S2 = 9.80665
EARTH_STD_ATMOS_SEA_LEVEL_PRESSURE_PA = 1.013250e5
EARTH_STD_ATMOS_SEA_LEVEL_TEMPERATURE_K = 288.15
EARTH_STD_ATMOS_SEA_LEVEL_DENSITY_KG_M3 = 1.225
EARTH_STD_ATMOS_SEA_LEVEL_DYNAMIC_VISCOSITY_NS_M2 = 1.7894e-5
EARTH_STD_ATMOS_SEA_LEVEL_MEAN_MOLECULAR_WEIGHT_KG_KMOL = 28.9644
EARTH_RADIUS_M = 6.356766e6

MAX_VALID_GEOPOTENTIAL_ALTITUDE_M = 84500.0
MAX_VALID_GEOMETRIC_ALTITUDE_M = 85638.0

GEOPOTENTIAL_ALTITUDE_LAYERS_TEMPERATURE_GRADIENT_K_M = {
    0.0: -6.5e-3,
    11000.0: 0.0,
    20000.0: 1.0e-3,
    32000.0: 2.8e-3,
    47000.0: 0.0,
    51000.0: -2.8e-3,
    71000.0: -2.0e-3,
}

GEOPOTENTIAL_ALTITUDE_LAYERS_INITIAL_TEMPERATURE_K = {
    0.0: EARTH_STD_ATMOS_SEA_LEVEL_TEMPERATURE_K,
    11000.0: 216.65,
    20000.0: 216.65,
    32000.0: 228.65,
    47000.0: 270.65,
    51000.0: 270.65,
    71000.0: 214.65,
    84500.0: 187.65,
}

GEOPOTENTIAL_ALTITUDE_LAYERS_INITIAL_PRESSURE_PA = {
    0.0: EARTH_STD_ATMOS_SEA_LEVEL_PRESSURE_PA,
    11000.0: 2.2632e4,
    20000.0: 5.4748e3,
    32000.0: 8.6801e2,
    47000.0: 1.1090e2,
    51000.0: 6.6938e1,
    71000.0: 3.9564e0,
    84500.0: 3.9814e-1,
}


@dataclass
class AtmosState:
    geometric_altitude_m: float
    geopotential_altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    dynamic_viscosity_ns_m2: float
    sound_speed_m_s: float


def calculate_geopotential_altitude(geometric_altitude_m: float) -> float:
    """Given a geometric altitude in meters will return the corresponding
    geopotential altitude, also in meters.

    Args:
        geometric_altitude_m (float): geometric altitude, measured from sea level, in meters.

    Returns:
        float: corresponding geopotential altitude in meters.
    """

    r0 = EARTH_RADIUS_M
    Z = geometric_altitude_m

    geopotential_altitude_m = (r0 * Z) / (r0 + Z)

    return geopotential_altitude_m


def calculate_geometric_altitude(geopotential_altitude_m: float) -> float:
    """Given a geopotential altitude in meters will return the corresponding
    geometric altitude, also in meters.

    Args:
        geopotential_altitude_m (float): geopotential altitude, in meters.

    Returns:
        float: corresponding geometric altitude, measured from sea level, in meters.
    """

    r0 = EARTH_RADIUS_M
    H = geopotential_altitude_m

    geometric_altitude_m = (r0 * H) / (r0 - H)

    return geometric_altitude_m


def _get_base_altitude(geopotential_altitude_m: float) -> float:
    """Finds the base altitude for the integration layer.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.

    Returns:
        float: base altitude of the layer in meters.
    """
    base_altitude_m = 0.0
    for altitude_m in GEOPOTENTIAL_ALTITUDE_LAYERS_TEMPERATURE_GRADIENT_K_M.keys():
        if altitude_m > geopotential_altitude_m:
            break
        base_altitude_m = altitude_m
    return base_altitude_m


def calculate_std_atmos_temperature(geopotential_altitude_m: float) -> float:
    """Given a geopotential altitude in meters will calculate the standard atmosphere temperature
    for that altitude in Kelvin. Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.

    Returns:
        float: standard atmosphere temperature in Kelvin.
    """

    base_altitude_m = _get_base_altitude(geopotential_altitude_m)

    H = geopotential_altitude_m
    H_b = base_altitude_m
    T_mb = GEOPOTENTIAL_ALTITUDE_LAYERS_INITIAL_TEMPERATURE_K[base_altitude_m]
    L_mb = GEOPOTENTIAL_ALTITUDE_LAYERS_TEMPERATURE_GRADIENT_K_M[base_altitude_m]

    T_m = T_mb + L_mb * (H - H_b)

    return T_m


def calculate_std_atmos_pressure(geopotential_altitude_m: float) -> float:
    """Given a geopotential altitude in meters will calculate the standard atmosphere pressure
    for that altitude in Pascals. Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.

    Returns:
        float: standard atmosphere pressure in Pascals.
    """

    base_altitude_m = _get_base_altitude(geopotential_altitude_m)

    H = geopotential_altitude_m
    H_b = base_altitude_m
    T_mb = GEOPOTENTIAL_ALTITUDE_LAYERS_INITIAL_TEMPERATURE_K[base_altitude_m]
    L_mb = GEOPOTENTIAL_ALTITUDE_LAYERS_TEMPERATURE_GRADIENT_K_M[base_altitude_m]
    g_0 = EARTH_GRAVITY_ACCELERATION_M_S2
    R = IDEAL_GAS_CONSTANT_NM_KMOLK
    M_0 = EARTH_STD_ATMOS_SEA_LEVEL_MEAN_MOLECULAR_WEIGHT_KG_KMOL
    P_b = GEOPOTENTIAL_ALTITUDE_LAYERS_INITIAL_PRESSURE_PA[base_altitude_m]

    if L_mb == 0.0:
        num = -g_0 * M_0 * (H - H_b)
        den = R * T_mb
        P = P_b * exp(num / den)

    else:
        expoent = (g_0 * M_0) / (R * L_mb)
        num = T_mb
        den = T_mb + L_mb * (H - H_b)
        P = P_b * (num / den) ** expoent

    return P


def calculate_std_atmos_density(
    geopotential_altitude_m: float, temperature_offset_k: float = 0.0
) -> float:
    """Given a geopotential altitude in meters , and temperature offset in Kelvin
    will calculate the standard atmosphere density for that altitude in kg/m3.
    Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.
        temperature_offset_k (float, optional): temperature offset in relation to the
            standard day. Defaults to 0.0.

    Returns:
        float: standard atmosphere density in kg/m3
    """

    std_pressure_pa = calculate_std_atmos_pressure(geopotential_altitude_m)
    std_temperature_k = calculate_std_atmos_temperature(geopotential_altitude_m)

    P = std_pressure_pa
    T_m = std_temperature_k + temperature_offset_k
    R = IDEAL_GAS_CONSTANT_NM_KMOLK
    M_0 = EARTH_STD_ATMOS_SEA_LEVEL_MEAN_MOLECULAR_WEIGHT_KG_KMOL

    rho = (P * M_0) / (R * T_m)

    return rho


def calculate_std_atmos_dynamic_viscosity(
    geopotential_altitude_m: float, temperature_offset_k: float = 0.0
) -> float:
    """Given a geopotential altitude in meters , and temperature offset in Kelvin
    will calculate the standard atmosphere dynamic viscosity for that altitude in Ns/m2.
    Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.
        temperature_offset_k (float, optional): temperature offset in relation to the
            standard day. Defaults to 0.0.

    Returns:
        float: standard atmosphere dynamic viscosity in Ns/m2
    """

    std_temperature_k = calculate_std_atmos_temperature(geopotential_altitude_m)

    beta = 1.458e-6  # empirical constant in kg/(s.m.K^1/2)
    T = std_temperature_k + temperature_offset_k
    S = SUTHERLAND_CONSTANT_K

    mu = (beta * T ** (3 / 2)) / (T + S)

    return mu


def calculate_std_atmos_sound_speed(
    geopotential_altitude_m: float, temperature_offset_k: float = 0.0
) -> float:
    """Given a geopotential altitude in meters , and temperature offset in Kelvin
    will calculate the standard atmosphere speed of sound for that altitude in m/s.
    Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.
        temperature_offset_k (float, optional): temperature offset in relation to the
            standard day. Defaults to 0.0.

    Returns:
        float: standard atmosphere speed of sound in m/s
    """

    std_temperature_k = calculate_std_atmos_temperature(geopotential_altitude_m)

    T_m = std_temperature_k + temperature_offset_k
    R = IDEAL_GAS_CONSTANT_NM_KMOLK
    M_0 = EARTH_STD_ATMOS_SEA_LEVEL_MEAN_MOLECULAR_WEIGHT_KG_KMOL
    gamma = IDEAL_GAS_GAMMA_RATIO

    C_s = ((gamma * R * T_m) / M_0) ** (1 / 2)

    return C_s


def calculate_std_atmos(
    geometrical_altitude_m: float, temperature_offset_k: float = 0.0
) -> AtmosState:
    """Given a geometrical altitude, measured from sea level, in meters and a temperature
    offset in Kelvin, will return an object containing the standard atmosphere temperature,
    pressure, density, dynamic viscosity and speed of sound in S.I. units for that
    altitude and temperature offset.
    Valid from 0m to 84500m of geopotential altitude.

    Args:
        geometrical_altitude_m (float): geometrical altitude in meters
        temperature_offset_k (float, optional): temperature offset in relation to the
            standard day. Defaults to 0.0.
    Returns:
        AtmosState: object containing the corresponding STD atmosphere properties.
    """

    if (geometrical_altitude_m < 0.0) or (geometrical_altitude_m > MAX_VALID_GEOMETRIC_ALTITUDE_M):
        raise ValueError(
            f"The altitude {geometrical_altitude_m} is outside of the supported bounds."
            f"Valid values are from 0.0m to {MAX_VALID_GEOMETRIC_ALTITUDE_M:.1f}m"
        )

    geopotential_altitude_m = calculate_geopotential_altitude(geometrical_altitude_m)
    temperature_k = calculate_std_atmos_temperature(geopotential_altitude_m) + temperature_offset_k
    pressure_pa = calculate_std_atmos_pressure(geopotential_altitude_m)
    density_kg_m3 = calculate_std_atmos_density(geopotential_altitude_m, temperature_offset_k)
    dynamic_viscosity_ns_m2 = calculate_std_atmos_dynamic_viscosity(
        geopotential_altitude_m, temperature_offset_k
    )
    sound_speed_m_s = calculate_std_atmos_sound_speed(geopotential_altitude_m, temperature_offset_k)

    return AtmosState(
        geometric_altitude_m=geometrical_altitude_m,
        geopotential_altitude_m=geopotential_altitude_m,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        density_kg_m3=density_kg_m3,
        dynamic_viscosity_ns_m2=dynamic_viscosity_ns_m2,
        sound_speed_m_s=sound_speed_m_s,
    )
