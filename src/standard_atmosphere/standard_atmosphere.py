from dataclasses import dataclass
from math import exp
from typing import NamedTuple

AVOGADRO_CONSTANT_1_KMOL = 6.022169e26
BOLTZMANN_CONSTANT_NM_K = 1.380622e-23
IDEAL_GAS_CONSTANT_NM_KMOLK = 8.31432e3
IDEAL_GAS_GAMMA_RATIO = 1.4
SUTHERLAND_CONSTANT_K = 110
SUTHERLAND_VISCOSITY_COEFFICIENT = 1.458e-6  # kg/(s.m.K^1/2)
EARTH_GRAVITY_ACCELERATION_M_S2 = 9.80665
EARTH_STD_ATMOS_SEA_LEVEL_PRESSURE_PA = 1.013250e5
EARTH_STD_ATMOS_SEA_LEVEL_TEMPERATURE_K = 288.15
EARTH_STD_ATMOS_SEA_LEVEL_DENSITY_KG_M3 = 1.225
EARTH_STD_ATMOS_SEA_LEVEL_DYNAMIC_VISCOSITY_NS_M2 = 1.7894e-5
EARTH_STD_ATMOS_SEA_LEVEL_MEAN_MOLECULAR_WEIGHT_KG_KMOL = 28.9644
EARTH_RADIUS_M = 6.356766e6

MAX_VALID_GEOPOTENTIAL_ALTITUDE_M = 84500.0
MAX_VALID_GEOMETRIC_ALTITUDE_M = 85638.0


class Layer(NamedTuple):
    base_geopotential_altitude_m: float
    base_temperature_k: float
    base_pressure_pa: float
    temperature_gradient_k_m: float


ATMOSPHERE_LAYERS = [
    Layer(0.0, 288.15, 1.013250e5, -6.5e-3),
    Layer(11000.0, 216.65, 2.2632e4, 0.0),
    Layer(20000.0, 216.65, 5.4748e3, 1.0e-3),
    Layer(32000.0, 228.65, 8.6801e2, 2.8e-3),
    Layer(47000.0, 270.65, 1.1090e2, 0.0),
    Layer(51000.0, 270.65, 6.6938e1, -2.8e-3),
    Layer(71000.0, 214.65, 3.9564e0, -2.0e-3),
]


@dataclass(frozen=True)
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


def _get_layer(geopotential_altitude_m: float) -> Layer:
    """Finds the atmospheric layer for the given geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.

    Returns:
        Layer: The atmospheric layer properties.
    """
    target_layer = ATMOSPHERE_LAYERS[0]
    for layer in ATMOSPHERE_LAYERS:
        if layer.base_geopotential_altitude_m > geopotential_altitude_m:
            break
        target_layer = layer
    return target_layer


def calculate_density(pressure_pa: float, temperature_k: float) -> float:
    """Calculates density from pressure and temperature using Ideal Gas Law."""
    R = IDEAL_GAS_CONSTANT_NM_KMOLK
    M_0 = EARTH_STD_ATMOS_SEA_LEVEL_MEAN_MOLECULAR_WEIGHT_KG_KMOL
    return (pressure_pa * M_0) / (R * temperature_k)


def calculate_viscosity(temperature_k: float) -> float:
    """Calculates dynamic viscosity using Sutherland's Law."""
    beta = SUTHERLAND_VISCOSITY_COEFFICIENT
    S = SUTHERLAND_CONSTANT_K
    return (beta * temperature_k ** (3 / 2)) / (temperature_k + S)


def calculate_speed_of_sound(temperature_k: float) -> float:
    """Calculates speed of sound for an ideal gas."""
    R = IDEAL_GAS_CONSTANT_NM_KMOLK
    M_0 = EARTH_STD_ATMOS_SEA_LEVEL_MEAN_MOLECULAR_WEIGHT_KG_KMOL
    gamma = IDEAL_GAS_GAMMA_RATIO
    return ((gamma * R * temperature_k) / M_0) ** 0.5


def calculate_std_atmos_temperature(geopotential_altitude_m: float) -> float:
    """Given a geopotential altitude in meters will calculate the standard atmosphere temperature
    for that altitude in Kelvin. Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.

    Returns:
        float: standard atmosphere temperature in Kelvin.
    """
    layer = _get_layer(geopotential_altitude_m)

    H = geopotential_altitude_m
    H_b = layer.base_geopotential_altitude_m
    T_mb = layer.base_temperature_k
    L_mb = layer.temperature_gradient_k_m

    return T_mb + L_mb * (H - H_b)


def calculate_std_atmos_pressure(geopotential_altitude_m: float) -> float:
    """Given a geopotential altitude in meters will calculate the standard atmosphere pressure
    for that altitude in Pascals. Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.

    Returns:
        float: standard atmosphere pressure in Pascals.
    """
    layer = _get_layer(geopotential_altitude_m)

    H = geopotential_altitude_m
    H_b = layer.base_geopotential_altitude_m
    T_mb = layer.base_temperature_k
    L_mb = layer.temperature_gradient_k_m
    P_b = layer.base_pressure_pa

    g_0 = EARTH_GRAVITY_ACCELERATION_M_S2
    R = IDEAL_GAS_CONSTANT_NM_KMOLK
    M_0 = EARTH_STD_ATMOS_SEA_LEVEL_MEAN_MOLECULAR_WEIGHT_KG_KMOL

    if L_mb == 0.0:
        num = -g_0 * M_0 * (H - H_b)
        den = R * T_mb
        return P_b * exp(num / den)
    else:
        exponent = (g_0 * M_0) / (R * L_mb)
        num = T_mb
        den = T_mb + L_mb * (H - H_b)
        return P_b * (num / den) ** exponent


def calculate_std_atmos_density(
    geopotential_altitude_m: float, temperature_offset_k: float = 0.0
) -> float:
    """Given a geopotential altitude in meters, and temperature offset in Kelvin
    will calculate the standard atmosphere density for that altitude in kg/m3.
    Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.
        temperature_offset_k (float, optional): temperature offset in relation to the
            standard day. Defaults to 0.0.

    Returns:
        float: standard atmosphere density in kg/m3
    """
    pressure_pa = calculate_std_atmos_pressure(geopotential_altitude_m)
    temperature_k = (
        calculate_std_atmos_temperature(geopotential_altitude_m) + temperature_offset_k
    )
    return calculate_density(pressure_pa, temperature_k)


def calculate_std_atmos_dynamic_viscosity(
    geopotential_altitude_m: float, temperature_offset_k: float = 0.0
) -> float:
    """Given a geopotential altitude in meters, and temperature offset in Kelvin
    will calculate the standard atmosphere dynamic viscosity for that altitude in Ns/m2.
    Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.
        temperature_offset_k (float, optional): temperature offset in relation to the
            standard day. Defaults to 0.0.

    Returns:
        float: standard atmosphere dynamic viscosity in Ns/m2
    """
    temperature_k = (
        calculate_std_atmos_temperature(geopotential_altitude_m) + temperature_offset_k
    )
    return calculate_viscosity(temperature_k)


def calculate_std_atmos_sound_speed(
    geopotential_altitude_m: float, temperature_offset_k: float = 0.0
) -> float:
    """Given a geopotential altitude in meters, and temperature offset in Kelvin
    will calculate the standard atmosphere speed of sound for that altitude in m/s.
    Valid from 0m to 84500m of geopotential altitude.

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters.
        temperature_offset_k (float, optional): temperature offset in relation to the
            standard day. Defaults to 0.0.

    Returns:
        float: standard atmosphere speed of sound in m/s
    """
    temperature_k = (
        calculate_std_atmos_temperature(geopotential_altitude_m) + temperature_offset_k
    )
    return calculate_speed_of_sound(temperature_k)


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
    if (geometrical_altitude_m < 0.0) or (
        geometrical_altitude_m > MAX_VALID_GEOMETRIC_ALTITUDE_M
    ):
        raise ValueError(
            f"The altitude {geometrical_altitude_m} is outside of the supported bounds."
            f"Valid values are from 0.0m to {MAX_VALID_GEOMETRIC_ALTITUDE_M:.1f}m"
        )

    geopotential_altitude_m = calculate_geopotential_altitude(geometrical_altitude_m)

    # Calculate state variables once
    temperature_k = calculate_std_atmos_temperature(geopotential_altitude_m)
    pressure_pa = calculate_std_atmos_pressure(geopotential_altitude_m)

    # Apply offset
    final_temperature_k = temperature_k + temperature_offset_k

    # Calculate derived properties using pure physics functions
    density_kg_m3 = calculate_density(pressure_pa, final_temperature_k)
    dynamic_viscosity_ns_m2 = calculate_viscosity(final_temperature_k)
    sound_speed_m_s = calculate_speed_of_sound(final_temperature_k)

    return AtmosState(
        geometric_altitude_m=geometrical_altitude_m,
        geopotential_altitude_m=geopotential_altitude_m,
        temperature_k=final_temperature_k,
        pressure_pa=pressure_pa,
        density_kg_m3=density_kg_m3,
        dynamic_viscosity_ns_m2=dynamic_viscosity_ns_m2,
        sound_speed_m_s=sound_speed_m_s,
    )
