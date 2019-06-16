from hashlib import md5

from aiida.orm import Data
from aiida.plugins.entry_point import load_entry_point, get_entry_point_names


class EmpiricalPotential(Data):
    """
    Store the empirical potential data
    """
    entry_name = 'lammps.potentials'
    potential_filename = 'potential.pot'
    pot_lines_fname = 'potential_lines.txt'

    @classmethod
    def list_types(cls):
        return get_entry_point_names(cls.entry_name)

    @classmethod
    def load_type(cls, entry_name):
        return load_entry_point(cls.entry_name, entry_name)

    def __init__(self, **kwargs):

        structure = kwargs.pop('structure', None)
        kind_names = kwargs.pop('kind_names', None)
        potential_type = kwargs.pop('type', None)
        potential_data = kwargs.pop('data', None)

        super(EmpiricalPotential, self).__init__(**kwargs)

        self._set_kind_names(structure, kind_names)
        self.set_data(potential_type, potential_data)

    def _set_kind_names(self, structure=None, kind_names=None):
        if structure is not None and kind_names is not None:
            raise ValueError("only one of 'structure' or 'kind_names' must be provided")
        elif structure is not None:
            names = [site.name for site in structure.kinds]
            self.set_attribute('kind_names', names)
        elif kind_names is not None:
            self.set_attribute('kind_names', kind_names)
        else:
            raise ValueError("one of 'structure' or 'kind_names' must be provided")

    def set_data(self, potential_type, data=None):
        """
        Store the potential type (ex. Tersoff, EAM, LJ, ..)
        """
        if potential_type is None:
            raise ValueError("'potential_type' must be provided")
        if potential_type not in self.list_types():
            raise ValueError("'potential_type' must be in: {}".format(self.list_types()))
        module = self.load_type(potential_type)

        atom_style = module.ATOM_STYLE
        default_units = module.DEFAULT_UNITS

        data = {} if data is None else data
        pot_contents = module.generate_LAMMPS_potential(data)
        pot_lines = module.get_input_potential_lines(
            data, names=self.kind_names, potential_filename=self.potential_filename)

        self.set_attribute("potential_type", potential_type)
        self.set_attribute("atom_style", atom_style)
        self.set_attribute("default_units", default_units)

        if pot_contents is not None:
            self.set_attribute('potential_md5', md5(pot_contents.encode("utf-8")).hexdigest())
            with self.open(self.potential_filename, mode="w") as handle:
                handle.write(pot_contents)
        elif self.potential_filename in self.list_object_names():
            self.delete_object(self.potential_filename)

        self.set_attribute('input_lines_md5', md5(pot_lines.encode("utf-8")).hexdigest())
        with self.open(self.pot_lines_fname, mode="w") as handle:
            handle.write(pot_lines)

    @property
    def kind_names(self):
        return self.get_attribute('kind_names')

    @property
    def potential_type(self):
        return self.get_attribute('potential_type')

    @property
    def atom_style(self):
        """get lammps atom style
        """
        return self.get_attribute('atom_style')

    @property
    def default_units(self):
        return self.get_attribute('default_units')

    def get_potential_file(self):
        if self.potential_filename in self.list_object_names():
            return self.get_object_content(self.potential_filename, 'r')
        return None

    def get_input_potential_lines(self):
        """
        get the string command to put in lammps input to setup the potential
        Ex:
             pair_style      eam
             pair_coeff      * *  Si
        """
        return self.get_object_content(self.pot_lines_fname, 'r')
