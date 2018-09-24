from aiida.common.exceptions import InputValidationError
from aiida.common.utils import classproperty
from aiida.orm.calculation.job import JobCalculation
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, join_keywords
from aiida_lammps.validation import validate_with_json


def generate_LAMMPS_input(parameters_data,
                          potential_obj,
                          structure_file='data.gan',
                          trajectory_file='path.lammpstrj'):
    names_str = ' '.join(potential_obj._names)

    parameters = parameters_data.get_dict()

    lammps_date = convert_date_string(parameters.get("lammps_version", None))

    lammps_input_file = 'units           {0}\n'.format(
        potential_obj.default_units)
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      {0}\n'.format(
        potential_obj.atom_style)
    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    # TODO find exact version when changes were made
    if lammps_date <= convert_date_string('11 Nov 2013'):
        lammps_input_file += 'compute         stpa all stress/atom\n'
    else:
        lammps_input_file += 'compute         stpa all stress/atom NULL\n'

        #  xx,       yy,        zz,       xy,       xz,       yz
    lammps_input_file += 'compute         stgb all reduce sum c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]\n'
    lammps_input_file += 'variable        pr equal -(c_stgb[1]+c_stgb[2]+c_stgb[3])/(3*vol)\n'
    lammps_input_file += 'thermo_style    custom step temp press v_pr etotal c_stgb[1] c_stgb[2] c_stgb[3] c_stgb[4] c_stgb[5] c_stgb[6]\n'
    lammps_input_file += 'thermo_modify   norm no\n'  # don't normalize extensive quantities by the number of atoms
    # NB: norm no is the default for metal and real, but not for lj

    lammps_input_file += 'run 0\n'

    lammps_input_file += 'print           "$(xlo) $(xhi) $(xy)"\n'
    lammps_input_file += 'print           "$(ylo) $(yhi) $(xz)"\n'
    lammps_input_file += 'print           "$(zlo) $(zhi) $(yz)"\n'

    return lammps_input_file


class SinglePointCalculation(BaseLammpsCalculation, JobCalculation):

    _OUTPUT_FILE_NAME = 'log.lammps'
    _OUTPUT_TRAJECTORY_FILE_NAME = 'path.lammpstrj'

    def _init_internal_params(self):
        super(SinglePointCalculation, self)._init_internal_params()

        self._default_parser = 'lammps.single'

        self._retrieve_list = [self._OUTPUT_FILE_NAME]
        self._retrieve_temporary_list = [self._INPUT_UNITS]
        self._generate_input_function = generate_LAMMPS_input

    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        """
        retdict = JobCalculation._use_methods
        retdict.update(BaseLammpsCalculation._baseclass_use_methods)

        return retdict

    def validate_parameters(self, param_data, potential_object):
        if param_data is None:
            raise InputValidationError("parameter data not set")
        validate_with_json(param_data.get_dict(), "single")

        # ensure the potential and paramters are in the same unit systems
        # TODO convert between unit systems (e.g. using https://pint.readthedocs.io)
        # punits = param_data.get_dict()['units']
        # if not punits == potential_object.default_units:
        #     raise InputValidationError('the units of the parameters ({}) and potential ({}) are different'.format(
        #         punits, potential_object.default_units
        #     ))

        return True


# $MPI -n $NSLOTS $LAMMPS -sf gpu -pk gpu 2 neigh no -in in.md_data
