import numpy as np
import desisurvey.tiles


def donefrac_in_conditions(condnexp, configfn=None):
    """Get donefrac corresponding to number of tiles observed
    in different conditions.

    condnexp from desisurvey.scripts.collect_etc.number_in_conditions
    """
    import desisurvey.config
    config = desisurvey.config.Configuration(configfn)
    tiles = desisurvey.tiles.get_tiles()
    out = np.zeros(tiles.ntiles, dtype=[
        ('TILEID', 'i4'),
        ('DONEFRAC_DARK', 'f4'), ('DONEFRAC_GRAY', 'f4'),
        ('DONEFRAC_BRIGHT', 'f4'),
        ('NNIGHT_DARK', 'f4'), ('NNIGHT_GRAY', 'f4'),
        ('NNIGHT_BRIGHT', 'f4'),
        ('NNIGHT_NEEDED_DARK', 'i4'), ('NNIGHT_NEEDED_GRAY', 'i4'),
        ('NNIGHT_NEEDED_BRIGHT', 'i4'),
    ])
    if ((len(np.unique(condnexp['TILEID'])) != len(condnexp)) or
            (len(np.unique(tiles.tileID)) != tiles.ntiles)):
        raise ValueError('Must be at least one tile per record!')
    _, mc, mt = np.intersect1d(condnexp['TILEID'], tiles.tileID,
                               return_indices=True)
    CONDITIONS = ['DARK', 'GRAY', 'BRIGHT']
    out['TILEID'] = tiles.tileID
    for i, program in enumerate(tiles.tileprogram):
        for cond in CONDITIONS:
            needed = getattr(config.completeness, program, 0)
            if not isinstance(needed, int):
                needed = getattr(needed, cond, None)
                needed = 0 if needed is None else needed()
            out['NNIGHT_NEEDED_'+cond][i] = needed
    for ic, it in zip(mc, mt):
        for cond in CONDITIONS:
            nexp = condnexp['NNIGHT_'+cond][ic]
            program = tiles.tileprogram[it]
            out['NNIGHT_'+cond][it] = nexp
    for cond in CONDITIONS:
        out['DONEFRAC_'+cond] = out['NNIGHT_'+cond]/(
            out['NNIGHT_NEEDED_'+cond] + (out['NNIGHT_NEEDED_'+cond] == 0))
    return out

    # we then output for each tile what conditions it's still allowed in,
    # and whether it's done (allowed_in_conditions = 0).

    # would be straightforward to incorporate completeness information
    # into optimize.py with an additional ~donefrac data structure
    # that modifies dlst_nom, multiplying it by (1-donefrac) for each tile

    # that also needs to know the donefrac in each condition.
    # but first step is just to get the donefracs at all.

    # Build the list of nexp for each of these conditions.
