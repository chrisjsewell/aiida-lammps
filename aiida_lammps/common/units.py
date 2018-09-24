_UNIT_STYLES = {
    'real': {
        'mass': 'grams/mole',
        'distance': 'angstroms',
        'time': 'femtoseconds',
        'energy': 'kcal/mole',
        'velocity': 'angstroms/femtosecond',
        'force': 'kcal/(mole*angstrom)',
        'torque': 'kcal/mole',
        'temperature': 'kelvin',
        'pressure': 'atmospheres',
        'dynamic_viscosity': 'poise',
        'charge': 'e',  # multiple of electron charge (1.0 is a proton)
        'dipole': 'charge*angstroms',
        'electric field': 'volts/angstrom',
        'density': 'gram/cm^3',
    },
    'metal': {
        'mass': 'grams/mole',
        'distance': 'angstroms',
        'time': 'picoseconds',
        'energy': 'eV',
        'velocity': 'angstroms/picosecond',
        'force': 'eV/angstrom',
        'torque': 'eV',
        'temperature': 'kelvin',
        'pressure': 'bars',
        'dynamic_viscosity': 'poise',
        'charge': 'e',  # multiple of electron charge (1.0 is a proton)
        'dipole': 'charge*angstroms',
        'electric field': 'volts/angstrom',
        'density': 'gram/cm^3',
    },
    'si': {
        'mass': 'kilograms',
        'distance': 'meters',
        'time': 'seconds',
        'energy': 'joules',
        'velocity': 'meters/second',
        'force': 'newtons',
        'torque': 'newton*meters',
        'temperature': 'kelvin',
        'pressure': 'pascals',
        'dynamic_viscosity': 'pascal*second',
        'charge': 'coulombs',  # (1.6021765e-19 is a proton)
        'dipole': 'coulombs*meters',
        'electric field': 'volts/meter',
        'density': 'kilograms/meter^3',
    },
    'cgs': {
        'mass': 'grams',
        'distance': 'centimeters',
        'time': 'seconds',
        'energy': 'ergs',
        'velocity': 'centimeters/second',
        'force': 'dynes',
        'torque': 'dyne*centimeters',
        'temperature': 'kelvin',
        'pressure': 'dyne/cm^2',  # or barye': '1.0e-6 bars
        'dynamic_viscosity': 'poise',
        'charge': 'statcoulombs',  # or esu (4.8032044e-10 is a proton)
        'dipole': 'statcoulombs*cm',  #: '10^18 debye
        'electric_field': 'statvolt/cm',  # or dyne/esu
        'density': 'grams/cm^3',
    },
    'electron': {
        'mass': 'amu',
        'distance': 'bohr',
        'time': 'femtoseconds',
        'energy': 'hartrees',
        'velocity': 'bohr/atu',  # [1.03275e-15 seconds]
        'force': 'hartrees/bohr',
        'temperature': 'kelvin',
        'pressure': 'pascals',
        'charge': 'e',  # multiple of electron charge (1.0 is a proton)
        'dipole_moment': 'debye',
        'electric_field': 'volts/cm',
    },
    'micro': {
        'mass': 'picograms',
        'distance': 'micrometers',
        'time': 'microseconds',
        'energy': 'picogram*micrometer^2/microsecond^2',
        'velocity': 'micrometers/microsecond',
        'force': 'picogram*micrometer/microsecond^2',
        'torque': 'picogram*micrometer^2/microsecond^2',
        'temperature': 'kelvin',
        'pressure': 'picogram/(micrometer*microsecond^2)',
        'dynamic_viscosity': 'picogram/(micrometer*microsecond)',
        'charge': 'picocoulombs',  # (1.6021765e-7 is a proton)
        'dipole': 'picocoulomb*micrometer',
        'electric field': 'volt/micrometer',
        'density': 'picograms/micrometer^3',
    },
    'nano': {
        'mass': 'attograms',
        'distance': 'nanometers',
        'time': 'nanoseconds',
        'energy': 'attogram*nanometer^2/nanosecond^2',
        'velocity': 'nanometers/nanosecond',
        'force': 'attogram*nanometer/nanosecond^2',
        'torque': 'attogram*nanometer^2/nanosecond^2',
        'temperature': 'kelvin',
        'pressure': 'attogram/(nanometer*nanosecond^2)',
        'dynamic_viscosity': 'attogram/(nanometer*nanosecond)',
        'charge': 'e',  # multiple of electron charge (1.0 is a proton)
        'dipole': 'charge*nanometer',
        'electric_field': 'volt/nanometer',
        'density': 'attograms/nanometer^3'
    }
}


def get_units_dict(style, quantities):
    """

    :param style: the unit style set in the lammps input
    :type style: str
    :param quantities: the quantities to get units for
    :type quantities: list of str
    :return:
    """
    out_dict = {}
    for quantity in quantities:
        out_dict[quantity + "_units"] = _UNIT_STYLES[style][quantity]
    return out_dict


def convert_units(value, inunits, outunits, per_atoms=None):
    """ convert value from one unit to another

    :type value: float
    :type inunits: str
    :type outunits: str
    :param per_atoms: required if converting between per mole and total quantities
    :type per_atoms: int
    :return:
    """
    import pint
    ureg = pint.UnitRegistry()

    if inunits == outunits:
        return value

    quantity = ureg.Quantity(value, inunits)

    inmoles = quantity.dimensionality.get('[substance]', 0)
    outmoles = ureg.Quantity(1, outunits).dimensionality.get('[substance]', 0)
    if inmoles - outmoles:
        if not per_atoms:
            raise ValueError(
                "'per mole' to/from 'total value' conversion required, but per_atoms not given"
            )
        quantity = quantity * (ureg.N_A / per_atoms)**(inmoles - outmoles)

    out_quantity = quantity.to(outunits)

    return out_quantity.magnitude
