import copy
import re
from decimal import Decimal

import numpy as np

# TODO can X be in middle of species?


def skip(f, numlines):
    for i in range(numlines):
        f.readline()


def readnumline(f, vtype):
    line = f.readline().split()
    if line[1] != '!':
        raise Exception(
            'not on a line containing ! (as expected) while reading {0}'.
            format(vtype))
    return int(line[0])


def split_numbers(string, as_decimal=False):
    """ get a list of numbers from a string (even with no spacing)

    Parameters
    ----------
    string: str
    as_decimal: bool
        if True return floats as decimal.Decimal objects

    Returns
    --------
    float_list: List

    Examples
    --------
    >>> split_numbers("1")
    [1.0]

    >>> split_numbers("1 2")
    [1.0, 2.0]

    >>> split_numbers("1.1 2.3")
    [1.1, 2.3]

    >>> split_numbers("1e-3")
    [0.001]

    >>> split_numbers("-1-2")
    [-1.0, -2.0]

    >>> split_numbers("1e-3-2")
    [0.001, -2.0]

    """
    _match_number = re.compile(
        '-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[+-]?\ *[0-9]+)?')
    string = string.replace(" .", " 0.")
    string = string.replace("-.", "-0.")
    return [
        Decimal(s) if as_decimal else float(s)
        for s in re.findall(_match_number, string)
    ]


def readlist(f):
    return split_numbers(f.readline())


def getvals(key, vals, names):
    out = []
    for name in names:
        out.append(vals[key.index(name)])
    return out


def transpose(lst):
    return list(map(list, zip(*lst)))


_paramkeys = [
    'Overcoordination 1', 'Overcoordination 2', 'Valency angle conjugation 1',
    'Triple bond stabilisation 1', 'Triple bond stabilisation 2',
    'C2-correction', 'Undercoordination 1', 'Triple bond stabilisation',
    'Undercoordination 2', 'Undercoordination 3',
    'Triple bond stabilization energy', 'Lower Taper-radius',
    'Upper Taper-radius', 'Not used 1', 'Valency undercoordination',
    'Valency angle/lone pair', 'Valency angle 1', 'Valency angle 2',
    'Not used 2', 'Double bond/angle', 'Double bond/angle: overcoord 1',
    'Double bond/angle: overcoord 2', 'Not used 3', 'Torsion/BO',
    'Torsion overcoordination 1', 'Torsion overcoordination 2', 'Not used 4',
    'Conjugation', 'vdWaals shielding', 'bond order cutoff',
    'Valency angle conjugation 2', 'Valency overcoordination 1',
    'Valency overcoordination 2', 'Valency/lone pair', 'Not used 5',
    'Not used 6', 'Not used 7', 'Not used 8', 'Valency angle conjugation 3'
]

_speckeys = [
    'idx', 'symbol', 'reaxff1_radii1', 'reaxff1_valence1', 'mass',
    'reaxff1_morse3', 'reaxff1_morse2', 'reaxff_gamma', 'reaxff1_radii2',
    'reaxff1_valence3', 'reaxff1_morse1', 'reaxff1_morse4', 'reaxff1_valence4',
    'reaxff1_under', 'dummy1', 'reaxff_chi', 'reaxff_mu', 'dummy2',
    'reaxff1_radii3', 'reaxff1_lonepair2', 'dummy3', 'reaxff1_over2',
    'reaxff1_over1', 'reaxff1_over3', 'dummy4', 'dummy5', 'reaxff1_over4',
    'reaxff1_angle1', 'dummy11', 'reaxff1_valence2', 'reaxff1_angle2',
    'dummy6', 'dummy7', 'dummy8'
]

_bondkeys = [
    'idx1', 'idx2', 'reaxff2_bond1', 'reaxff2_bond2', 'reaxff2_bond3',
    'reaxff2_bond4', 'reaxff2_bo5', 'reaxff2_bo7', 'reaxff2_bo6',
    'reaxff2_over', 'reaxff2_bond5', 'reaxff2_bo3', 'reaxff2_bo4', 'dummy1',
    'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo8', 'reaxff2_bo9'
]

_odkeys = [
    'idx1', 'idx2', 'reaxff2_morse1', 'reaxff2_morse3', 'reaxff2_morse2',
    'reaxff2_morse4', 'reaxff2_morse5', 'reaxff2_morse6'
]

_anglekeys = [
    'idx1', 'idx2', 'idx3', 'reaxff3_angle1', 'reaxff3_angle2',
    'reaxff3_angle3', 'reaxff3_conj', 'reaxff3_angle5', 'reaxff3_penalty',
    'reaxff3_angle4'
]

_torkeys = [
    'idx1', 'idx2', 'idx3', 'idx4', 'reaxff4_torsion1', 'reaxff4_torsion2',
    'reaxff4_torsion3', 'reaxff4_torsion4', 'reaxff4_torsion5', 'dummy1',
    'dummy2'
]

_hbkeys = [
    'idx1', 'idx2', 'idx3', 'reaxff3_hbond1', 'reaxff3_hbond2',
    'reaxff3_hbond3', 'reaxff3_hbond4'
]

_tolerance_defaults = {
    "anglemin": 0.001,
    "angleprod": 0.000001,
    "hbondmin": 0.01,
    "hbonddist": 7.5,
    "torsionprod":
    0.000000001  # NB: needs to be lower to get comparable energy to lammps, but then won't optimize
}


def read_reaxff_file(inpath, reaxfftol=None):
    """

    :param inpath: path to reaxff file (in standard (lammps) format)
    :param reaxfftol: additional tolerance parameters (required for gulp only)
    :return:
    """

    reaxfftol = {} if reaxfftol is None else reaxfftol.copy()
    toldict = {}
    for key, val in _tolerance_defaults.items():
        if key in reaxfftol:
            toldict[key] = reaxfftol[key]
        else:
            toldict[key] = val

    with open(inpath, 'r') as f:
        # Descript Initial Line
        descript = f.readline()
        # Read Parameters
        # ----------
        npar = int(f.readline().split()[0])
        if npar != len(_paramkeys):
            raise IOError('Expecting {} general parameters'.format(
                len(_paramkeys)))
        reaxff_par = {}
        for i, key in enumerate(_paramkeys):
            reaxff_par[key] = float(f.readline().split()[0])

        # Read Species Information
        # --------------------
        nspec = readnumline(f, 'species')
        skip(f, 3)
        spec_values = []
        for i in range(nspec):
            values = f.readline().split()
            symbol = values.pop(0)
            idx = 0 if symbol == 'X' else i + 1
            spec_values.append([idx, symbol] + [float(v) for v in values] +
                               readlist(f) + readlist(f) + readlist(f))
            if len(spec_values[i]) != len(_speckeys):
                raise Exception(
                    'number of values different than expected for species {0}'.
                    format(symbol))

        # spec_df = pd.DataFrame(spec_values, columns=speckey).set_index('idx')
        # spec_df['reaxff1_lonepair1'] = 0.5 * (spec_df.reaxff1_valence3 - spec_df.reaxff1_valence1)

        spec_dict = {k: v for k, v in zip(_speckeys, transpose(spec_values))}
        spec_dict['reaxff1_lonepair1'] = (
            0.5 * (np.array(spec_dict["reaxff1_valence3"]) - np.array(
                spec_dict["reaxff1_valence1"]))).tolist()

        # Read Bond Information
        # --------------------
        # bondcode = ['idx1', 'idx2', 'Edis1', 'Edis2', 'Edis3',
        #             'pbe1', 'pbo5', '13corr', 'pbo6', 'kov',
        #             'pbe2', 'pbo3', 'pbo4', 'nu', 'pbo1', 'pbo2',
        #             'ovcorr', 'nu']
        # bonddescript = ['idx1', 'idx2', 'Sigma-bond dissociation energy', 'Pi-bond dissociation energy',
        #                 'Double pi-bond dissociation energy',
        #                 'Bond energy', 'Double pi bond order', '1,3-Bond order correction', 'Double pi bond order',
        #                 'Overcoordination penalty',
        #                 'Bond energy', 'Pi bond order', 'Pi bond order', 'dummy', 'Sigma bond order',
        #                 'Sigma bond order',
        #                 'Overcoordination BO correction', 'dummy']
        # bond_lookup_df = pd.DataFrame(np.array([bondcode, bonddescript]).T, index=bondkey)

        nbond = readnumline(f, 'bonds')
        skip(f, 1)
        bond_values = []
        for i in range(nbond):
            values = readlist(f)
            id1 = values.pop(0)
            id2 = values.pop(0)
            bond_values.append([int(id1), int(id2)] + values + readlist(f))
            if len(bond_values[i]) != len(_bondkeys):
                raise Exception(
                    'number of values different than expected for bond')
        # bond_df = pd.DataFrame(bond_values, columns=bondkey).set_index(['idx1', 'idx2'])
        bond_dict = {k: v for k, v in zip(_bondkeys, transpose(bond_values))}

        # Read Off-Diagonal Information
        # --------------------
        nod = readnumline(f, 'off-diagonal')
        od_values = []
        for i in range(nod):
            values = readlist(f)
            id1 = int(values.pop(0))
            id2 = int(values.pop(0))
            od_values.append([id1, id2] + values)
            if len(od_values[i]) != len(_odkeys):
                raise Exception(
                    'number of values different than expected for off-diagonal'
                )
        # od_df = pd.DataFrame(od_values, columns=odkey).set_index(['idx1', 'idx2'])
        od_dict = {k: v for k, v in zip(_odkeys, transpose(od_values))}

        # Read Angle Information
        # --------------------
        nangle = readnumline(f, 'angle')
        angle_values = []
        for i in range(nangle):
            values = readlist(f)
            id1 = int(values.pop(0))
            id2 = int(values.pop(0))
            id3 = int(values.pop(0))
            angle_values.append([id1, id2, id3] + values)
            if len(angle_values[i]) != len(_anglekeys):
                raise Exception(
                    'number of values different than expected for angle')
        # angle_df = pd.DataFrame(angle_values, columns=anglekey).set_index(['idx1', 'idx2', 'idx3'])
        angle_dict = {
            k: v
            for k, v in zip(_anglekeys, transpose(angle_values))
        }

        # Read Torsion Information
        # --------------------
        ntors = readnumline(f, 'torsion')
        torsion_values = []
        for i in range(ntors):
            values = readlist(f)
            species1 = int(values.pop(0))
            species2 = int(values.pop(0))
            species3 = int(values.pop(0))
            species4 = int(values.pop(0))
            torsion_values.append([species1, species2, species3, species4] +
                                  values)
            if len(torsion_values[i]) != len(_torkeys):
                raise Exception(
                    'number of values different than expected for torsion')
        # torsion_df = pd.DataFrame(torsion_values, columns=torkey).set_index(['idx1', 'idx2', 'idx3', 'idx4'])
        torsion_dict = {
            k: v
            for k, v in zip(_torkeys, transpose(torsion_values))
        }

        # Read HBond Information
        # --------------------
        nhb = readnumline(f, 'hbond')
        hbond_values = []
        for i in range(nhb):
            values = readlist(f)
            species1 = int(values.pop(0))
            species2 = int(values.pop(0))
            species3 = int(values.pop(0))
            hbond_values.append([species1, species2, species3] + values)
            if len(hbond_values[i]) != len(_hbkeys):
                raise Exception(
                    'number of values different than expected for hbond {0},{1},{2}'.
                    format(species1, species2, species3))
        # hbond_df = pd.DataFrame(hbond_values, columns=hbkey).set_index(['idx1', 'idx2', 'idx3'])
        hbond_dict = {k: v for k, v in zip(_hbkeys, transpose(hbond_values))}

        return {
            "descript": descript.strip(),
            "params": reaxff_par,
            "species":
            spec_dict,  # spec_df.reset_index().to_dict(orient='list'),
            "bonds":
            bond_dict,  # bond_df.reset_index().to_dict(orient='list'),
            "off-diagonals":
            od_dict,  # od_df.reset_index().to_dict(orient='list'),
            "hbonds":
            hbond_dict,  # hbond_df.reset_index().to_dict(orient='list'),
            "torsions":
            torsion_dict,  # torsion_df.reset_index().to_dict(orient='list'),
            "angles":
            angle_dict,  # angle_df.reset_index().to_dict(orient='list')
        }


def write_lammps(data):
    """write reaxff data in GULP input format

    :param data: dictionary of data
    :rtype: str
    """
    outstr = ""
    regex = lambda x: " ".join(["{:.4f}"] * x)

    outstr += ("{}".format(data["descript"]))
    # if species_filter:
    #     outstr += ("#  (Filtered by: {})\n".format(species_filter))
    outstr += "\n"

    outstr += "{} ! Number of general parameters\n".format(len(_paramkeys))
    for key in _paramkeys:
        outstr += "{0:.4f} ! {1}\n".format(data["params"][key], key)

    outstr += '{0} ! Nr of atoms; cov.r; valency;a.m;Rvdw;Evdw;gammaEEM;cov.r2;#\n'.format(
        len(data["species"][_speckeys[0]]))
    outstr += 'alfa;gammavdW;valency;Eunder;Eover;chiEEM;etaEEM;n.u.\n'
    outstr += 'cov r3;Elp;Heat inc.;n.u.;n.u.;n.u.;n.u.\n'
    outstr += 'ov/un;val1;n.u.;val3,vval4\n'

    spec_data = transpose(
        [data['species'][key] for key in _speckeys if key != 'idx'])

    for spec in spec_data:
        outstr += spec[0] + ' ' + regex(8).format(*spec[1:9]) + '\n'
        outstr += regex(8).format(*spec[9:17]) + '\n'
        outstr += regex(8).format(*spec[17:25]) + '\n'
        outstr += regex(8).format(*spec[25:33]) + '\n'

    outstr += '{0} ! Nr of bonds; Edis1;LPpen;n.u.;pbe1;pbo5;13corr;pbo6\n'.format(
        len(data["bonds"][_bondkeys[0]]))
    outstr += 'pbe2;pbo3;pbo4;n.u.;pbo1;pbo2;ovcorr\n'

    bond_data = transpose([data['bonds'][key] for key in _bondkeys])

    for bond in bond_data:
        outstr += '{} {} '.format(
            bond[0], bond[1]) + regex(8).format(*bond[2:10]) + '\n'
        outstr += regex(8).format(*bond[10:18]) + '\n'

    outstr += '{0} ! Nr of off-diagonal terms; Ediss;Ro;gamma;rsigma;rpi;rpi2\n'.format(
        len(data["off-diagonals"][_odkeys[0]]))

    od_data = transpose([data['off-diagonals'][key] for key in _odkeys])

    for od in od_data:
        outstr += '{} {} '.format(od[0],
                                  od[1]) + regex(6).format(*od[2:8]) + '\n'

    outstr += '{0} ! Nr of angles;at1;at2;at3;Thetao,o;ka;kb;pv1;pv2\n'.format(
        len(data["angles"][_anglekeys[0]]))

    angle_data = transpose([data['angles'][key] for key in _anglekeys])

    for angle in angle_data:
        outstr += '{} {} {} '.format(*angle[0:3]) + regex(7).format(
            *angle[3:10]) + '\n'

    outstr += '{0} ! Nr of torsions;at1;at2;at3;at4;;V1;V2;V3;V2(BO);vconj;n.u;n\n'.format(
        len(data["torsions"][_torkeys[0]]))

    torsion_data = transpose([data['torsions'][key] for key in _torkeys])

    for tor in torsion_data:
        outstr += '{} {} {} {} '.format(*tor[0:4]) + regex(7).format(
            *tor[4:11]) + '\n'

    outstr += '{0} ! Nr of hydrogen bonds;at1;at2;at3;Rhb;Dehb;vhb1\n'.format(
        len(data["hbonds"][_hbkeys[0]]))

    hbond_data = transpose([data['hbonds'][key] for key in _hbkeys])

    for hbond in hbond_data:
        outstr += '{} {} {} '.format(*hbond[0:3]) + regex(4).format(
            *hbond[3:8]) + '\n'

    return outstr
