from aiida.orm.calculation.job import JobCalculation
from aiida.common.exceptions import InputValidationError
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.utils import classproperty
from aiida.orm import DataFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

from aiida_lammps.calculations.lammps.potentials import LammpsPotential
import numpy as np


def get_supercell(structure, supercell_shape):
    import itertools

    symbols = np.array([site.kind_name for site in structure.sites])
    positions = np.array([site.position for site in structure.sites])
    cell = np.array(structure.cell)
    supercell_shape = np.array(supercell_shape.dict.shape)

    supercell_array = np.dot(cell, np.diag(supercell_shape))

    supercell = StructureData(cell=supercell_array)
    for k in range(positions.shape[0]):
        for r in itertools.product(*[range(i) for i in supercell_shape[::-1]]):
            position = positions[k, :] + np.dot(np.array(r[::-1]), cell)
            symbol = symbols[k]
            supercell.append_atom(position=position, symbols=symbol)

    return supercell


def get_FORCE_CONSTANTS_txt(force_constants):

    force_constants = force_constants.get_array('force_constants')

    fc_shape = force_constants.shape
    fc_txt = "%4d\n" % (fc_shape[0])
    for i in range(fc_shape[0]):
        for j in range(fc_shape[1]):
            fc_txt += "%4d%4d\n" % (i + 1, j + 1)
            for vec in force_constants[i][j]:
                fc_txt += ("%22.15f" * 3 + "\n") % tuple(vec)

    return fc_txt


def structure_to_poscar(structure):

    atom_type_unique = np.unique(
        [site.kind_name for site in structure.sites], return_index=True)[1]
    labels = np.diff(np.append(atom_type_unique, [len(structure.sites)]))

    poscar = ' '.join(np.unique([site.kind_name for site in structure.sites]))
    poscar += '\n1.0\n'
    cell = structure.cell
    for row in cell:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*row)
    poscar += ' '.join(
        np.unique([site.kind_name for site in structure.sites])) + '\n'
    poscar += ' '.join(np.array(labels, dtype=str)) + '\n'
    poscar += 'Cartesian\n'
    for site in structure.sites:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(
            *site.position)

    return poscar


def parameters_to_input_file(parameters_object):

    parameters = parameters_object.get_dict()
    input_file = ('STRUCTURE FILE POSCAR\nPOSCAR\n\n')
    input_file += ('FORCE CONSTANTS\nFORCE_CONSTANTS\n\n')
    input_file += ('PRIMITIVE MATRIX\n')
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[0])
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[1])
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[2])
    input_file += ('\n')
    input_file += ('SUPERCELL MATRIX PHONOPY\n')
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[0])
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[1])
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[2])
    input_file += ('\n')

    return input_file


def generate_LAMMPS_structure(structure, atom_style):
    import numpy as np

    types = [site.kind_name for site in structure.sites]

    type_index_unique = np.unique(types, return_index=True)[1]
    count_index_unique = np.diff(np.append(type_index_unique, [len(types)]))

    atom_index = []
    for i, index in enumerate(count_index_unique):
        atom_index += [i for j in range(index)]

    masses = [site.mass for site in structure.kinds]
    positions = [site.position for site in structure.sites]

    number_of_atoms = len(positions)

    lammps_data_file = 'Generated using dynaphopy\n\n'
    lammps_data_file += '{0} atoms\n\n'.format(number_of_atoms)
    lammps_data_file += '{0} atom types\n\n'.format(len(masses))

    cell = np.array(structure.cell)

    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])

    alpha = np.arccos(np.dot(cell[1], cell[2]) / (c * b))
    gamma = np.arccos(np.dot(cell[1], cell[0]) / (a * b))
    beta = np.arccos(np.dot(cell[2], cell[0]) / (a * c))

    xhi = a
    xy = b * np.cos(gamma)
    xz = c * np.cos(beta)
    yhi = np.sqrt(pow(b, 2) - pow(xy, 2))
    yz = (b * c * np.cos(alpha) - xy * xz) / yhi
    zhi = np.sqrt(pow(c, 2) - pow(xz, 2) - pow(yz, 2))

    xhi = xhi + max(0, 0, xy, xz, xy + xz)
    yhi = yhi + max(0, 0, yz)

    lammps_data_file += '\n{0:20.10f} {1:20.10f} xlo xhi\n'.format(0, xhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} ylo yhi\n'.format(0, yhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} zlo zhi\n'.format(0, zhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f} xy xz yz\n\n'.format(
        xy, xz, yz)

    lammps_data_file += 'Masses\n\n'

    for i, mass in enumerate(masses):
        lammps_data_file += '{0} {1:20.10f} \n'.format(i + 1, mass)

    lammps_data_file += '\nAtoms\n\n'
    for i, row in enumerate(positions):
        if atom_style == 'charge':
            # TODO variable initial charge
            lammps_data_file += '{0} {1} 0.0 {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(
                i + 1, atom_index[i] + 1, row[0], row[1], row[2])
        else:
            lammps_data_file += '{0} {1} {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(
                i + 1, atom_index[i] + 1, row[0], row[1], row[2])

    return lammps_data_file


def generate_LAMMPS_potential(pair_style):

    potential_file = '# Potential file generated by aiida plugin (please check citation in the orignal file)\n'
    for key, value in pair_style.dict.data.iteritems():
        potential_file += '{}    {}\n'.format(key, value)

    return potential_file


class BaseLammpsCalculation(JobCalculation):
    """
    A basic plugin for calculating force constants using Lammps.

    Requirement: the node should be able to import phonopy
    """

    _INPUT_FILE_NAME = 'input.in'
    _INPUT_POTENTIAL = 'potential.pot'
    _INPUT_STRUCTURE = 'input.data'
    _INPUT_UNITS = 'input.units'

    _retrieve_list = []
    _retrieve_temporary_list = []
    _cmdline_params = ['-in', _INPUT_FILE_NAME]
    _stdout_name = None

    def _init_internal_params(self):
        super(BaseLammpsCalculation, self)._init_internal_params()

    @classproperty
    def _baseclass_use_methods(cls):
        """
        Common methods for LAMMPS.
        """

        retdict = {
            "potential": {
                'valid_types':
                ParameterData,
                'additional_parameter':
                None,
                'linkname':
                'potential',
                'docstring': ("Use a node that specifies the lammps potential "
                              "for the namelists"),
            },
            "structure": {
                'valid_types': StructureData,
                'additional_parameter': None,
                'linkname': 'structure',
                'docstring': "Use a node for the structure",
            },
            "parameters": {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'parameters',
                'docstring': "Use a node for the lammps input parameters",
            },
        }

        return retdict

    def _create_additional_files(self, tempfolder, inputs_params):
        pass

    def validate_parameters(self, param_data, potential_object):
        return True

    def _prepare_for_submission(self, tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """

        self._parameters_data = inputdict.pop(
            self.get_linkname('parameters'), None)

        try:
            potential_data = inputdict.pop(self.get_linkname('potential'))
        except KeyError:
            raise InputValidationError("No potential specified for this "
                                       "calculation")

        if not isinstance(potential_data, ParameterData):
            raise InputValidationError("potential is not of type "
                                       "ParameterData")

        try:
            self._structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise InputValidationError(
                "no structure is specified for this calculation")

        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError(
                "no code is specified for this calculation")

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        # =================== prepare the python input files =====================

        potential_object = LammpsPotential(
            potential_data,
            self._structure,
            potential_filename=self._INPUT_POTENTIAL)

        # validate the parameters
        self.validate_parameters(self._parameters_data, potential_object)

        # TODO is this the best way to tell the parser what units were used?
        # depending on how the calculation is run, it can be either stored or unstored at this point,
        # which prohibits using self.set_attr / self.set_extra / self._units =
        input_units_filename = tempfolder.get_abs_path(self._INPUT_UNITS)
        with open(input_units_filename, 'w') as infile:
            infile.write(potential_object.default_units)

        structure_txt = generate_LAMMPS_structure(self._structure,
                                                  potential_object.atom_style)
        input_txt = self._generate_input_function(
            self._parameters_data,
            potential_object,
            structure_file=self._INPUT_STRUCTURE,
            trajectory_file=self._OUTPUT_TRAJECTORY_FILE_NAME)

        potential_txt = potential_object.get_potential_file()

        # =========================== dump to file =============================

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)
        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        structure_filename = tempfolder.get_abs_path(self._INPUT_STRUCTURE)
        with open(structure_filename, 'w') as infile:
            infile.write(structure_txt)

        if potential_txt is not None:
            potential_filename = tempfolder.get_abs_path(self._INPUT_POTENTIAL)
            with open(potential_filename, 'w') as infile:
                # print(potential_txt)
                infile.write(potential_txt)

        self._create_additional_files(tempfolder, inputdict)

        # ============================ calcinfo ================================

        local_copy_list = []
        remote_copy_list = []
        # additional_retrieve_list = settings_dict.pop("ADDITIONAL_RETRIEVE_LIST",[])

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        # Empty command line by default
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        # Retrieve files
        calcinfo.retrieve_list = self._retrieve_list
        calcinfo.retrieve_temporary_list = self._retrieve_temporary_list
        codeinfo = CodeInfo()
        codeinfo.cmdline_params = self._cmdline_params
        codeinfo.code_uuid = code.uuid
        codeinfo.withmpi = False  # Set lammps openmpi environment properly
        calcinfo.codes_info = [codeinfo]
        codeinfo.stdout_name = self._stdout_name
        return calcinfo
