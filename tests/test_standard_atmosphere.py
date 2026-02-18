import pytest
import csv
from pathlib import Path
from standard_atmosphere.standard_atmosphere import calculate_std_atmos, MAX_VALID_GEOMETRIC_ALTITUDE_M

@pytest.fixture
def atmos_data():
    csv_path = Path(__file__).parent / "test_data.csv"
    data = []
    with open(csv_path, newline='') as csvfile:
        # Filter out comments and empty lines
        filtered_lines = (
            line for line in csvfile 
            if not line.startswith('#') and line.strip()
        )
        reader = csv.DictReader(filtered_lines)
        for row in reader:
            data.append(row)
    return data


def test_calculate_std_atmos_against_table(atmos_data):
    for row in atmos_data:
        alt_km = float(row['alt'])
        geo_alt_m = alt_km * 1000.0

        # Skip altitudes outside valid range
        if geo_alt_m > MAX_VALID_GEOMETRIC_ALTITUDE_M:
            continue

        res = calculate_std_atmos(geo_alt_m)

        # Temperature
        expected_temp = float(row['temp'])
        assert res.temperature_k == pytest.approx(expected_temp, rel=1e-2), \
            f"Temperature mismatch at {alt_km} km"

        # Pressure
        expected_press = float(row['press'])
        assert res.pressure_pa == pytest.approx(expected_press, rel=5e-2), \
            f"Pressure mismatch at {alt_km} km"

        # Density
        expected_dens = float(row['dens'])
        assert res.density_kg_m3 == pytest.approx(expected_dens, rel=1e-2), \
            f"Density mismatch at {alt_km} km"

        # Speed of Sound
        expected_sound_speed = float(row['a'])
        assert res.sound_speed_m_s == pytest.approx(expected_sound_speed, rel=1e-2), \
            f"Speed of sound mismatch at {alt_km} km"

        # Dynamic Viscosity
        # CSV has values like 17.89 which means 17.89 * 10^-6
        expected_visc = float(row['visc']) * 1e-6
        assert res.dynamic_viscosity_ns_m2 == pytest.approx(expected_visc, rel=1e-2), \
            f"Viscosity mismatch at {alt_km} km"
