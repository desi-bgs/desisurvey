"""Plan future DESI observations.
"""
from __future__ import print_function, division

import datetime
import os.path

import numpy as np

import astropy.table
import astropy.io.fits
import astropy.units as u

import desiutil.log

import desimodel.io

import desisurvey.utils


def load_design_hourangle(name='surveyinit.fits'):
    """Load design hour-angle assignments from disk.

    Reads column 'HA' from binary table HDU 'DESIGN'. This is the format
    saved by the ``surveyinit`` script, but any FITS file following the
    same convention can be used.

    Parameters
    ----------
    name : str
        Name of the FITS file to read. A relative path is assumed to
        refer to the output path specified in the configuration.
    
    Returns
    -------
    array
        1D array of design hour angles in degrees, with indexing
        that matches :class:`desisurvey.tiles.Tiles`.
    """
    config = desisurvey.config.Configuration()
    fullname = config.get_path(name)
    with astropy.io.fits.open(fullname, memmap=False) as hdus:
        HA = hdus['DESIGN'].data['HA'].copy()
    tiles = desisurvey.tiles.get_tiles()
    if HA.shape != (tiles.ntiles,):
        raise ValueError('Read unexpected HA shape.')
    return HA


def load_weather(start_date=None, stop_date=None, name='surveyinit.fits'):
    """Load dome-open fraction expected during each night of the survey.
    
    Reads Image HDU 'WEATHER'. This is the format saved by the
    ``surveyinit`` script, but any FITS file following the same
    convention can be used.

    Parameters
    ----------
    name : str
        Name of the FITS file to read. A relative path is assumed to
        refer to the output path specified in the configuration.
    start_date : date or None
        First night to include or use the first date of the survey. Must
        be convertible to a date using :func:`desisurvey.utils.get_date`.
    stop_date : date or None
        First night to include or use the last date of the survey. Must
        be convertible to a date using :func:`desisurvey.utils.get_date`.

    Returns
    -------
    array
        1D array of length equal to the span between stop_date and
        start_date. Values are between 0 (dome closed all night) and
        1 (dome open all night).
    """
    config = desisurvey.config.Configuration()
    if start_date is None:
        start_date = config.first_day()
    else:
        start_date = desisurvey.utils.get_date(start_date)
    if stop_date is None:
        stop_date = config.last_day()
    else:
        stop_date = desisurvey.utils.get_date(stop_date)
    if stop_date <= start_date:
        raise ValueError('Expected start_date < stop_date.')
    with astropy.io.fits.open(config.get_path(name), memmap=False) as hdus:
        weather = hdus['WEATHER'].data
        num_nights = len(weather)
        first = desisurvey.utils.get_date(hdus['WEATHER'].header['FIRST'])
        last = first + datetime.timedelta(num_nights)
    if start_date < first:
        raise ValueError('Weather not available before {}.'.format(first.isoformat()))
    num_nights = (stop_date - start_date).days
    if last < stop_date:
        raise ValueError('Weather not available after {}.',format(last.isoformat()))
    ilo, ihi = (start_date - first).days, (stop_date - first).days
    return weather[ilo:ihi]


class Planner(object):
    """Coordinate afternoon planning activities.

    Parameters
    ----------
    rules : object or None
        Object with an ``apply`` method that is used to implement survey strategy by updating
        tile priorities each afternoon.  When None, all tiles have equal priority.
    fiberassign_cadence : 'daily' or 'monthly'
        Cadence for updating fiber assignments.  Monthly is defined as the afternoon before
        a full moon.
    restore_date : datetime.date or None
        Restore internal state from the snapshot saved on this date, or initialize a new
        plan when None.
    tiles_file : str or None
        Override the default tiles files specified in the configuration when specified.
    """
    def __init__(self, rules=None, fiberassign_cadence='monthly', restore_date=None, tiles_file=None):
        self.log = desiutil.log.get_logger()
        self.rules = rules
        if fiberassign_cadence not in ('daily', 'monthly'):
            raise ValueError('Invalid fiberassign_cadence: "{}".'.format(fiberassign_cadence))
        self.fiberassign_cadence = fiberassign_cadence
        config = desisurvey.config.Configuration()
        self.tiles = desisurvey.tiles.get_tiles(tiles_file)
        self.ephem = desisurvey.ephem.get_ephem()
        if restore_date is not None:
            name = config.get_path('plan_{}.fits'.format(restore_date.isoformat()))
            if not os.path.exists(name):
                raise RuntimeError('Cannot restore from non-existant "{}".'.format(name))
            t = astropy.table.Table.read(name)
            if t.meta['CADENCE'] != self.fiberassign_cadence:
                raise ValueError('Fiberassign cadence mismatch.')
            self.first_night = desisurvey.utils.get_date(t.meta['FIRST'])
            self.last_night = desisurvey.utils.get_date(t.meta['LAST'])
            self.tile_covered = t['COVERED'].data.copy()
            self.tile_countdown = t['COUNTDOWN'].data.copy()
            self.tile_available = t['AVAILABLE'].data.copy()
            self.tile_priority = t['PRIORITY'].data.copy()
            self.log.info(
                'Restored plan with {} ({}) / {} tiles covered (available).'
                .format(np.count_nonzero(self.tile_covered),
                        np.count_nonzero(self.tile_available),
                        self.tiles.ntiles))
        else:
            # Initialize per-tile arrays.
            self.tile_covered = np.full(self.tiles.ntiles, -1)
            self.tile_countdown = np.full(self.tiles.ntiles, 1)
            self.tile_available = np.zeros(self.tiles.ntiles, bool)
            self.tile_priority = None
            self.first_night = self.last_night = None
        # Precompute the tile overlaps between passes needed to update fiber assignments.
        self.tile_over = {}
        self.overlapping = {}
        fiberassign_order = config.fiber_assignment_order
        tile_diameter = 2 * config.tile_radius().to(u.deg).value
        for passnum in self.tiles.passes:
            under = self.tiles.passnum == passnum
            over = np.zeros_like(under)
            key = 'P{}'.format(passnum)
            if key not in fiberassign_order.keys:
                if restore_date is None:
                    # Mark tiles in this pass as initially available.
                    self.tile_covered[under] = 0
                    self.tile_available[under] = True
                    self.log.info('Pass {} available for initial observing.'.format(passnum))
            else:
                overpasses = getattr(fiberassign_order, key)()
                for overpass in overpasses.split('+'):
                    if not len(overpass) == 2 and overpass[0] == 'P':
                        raise RuntimeError(
                            'Invalid pass in fiber_assignment_order: {}.'
                            .format(overpass))
                    over |= (self.tiles.passnum == int(overpass[1]))
                self.overlapping[passnum] = desisurvey.utils.separation_matrix(
                    self.tiles.tileRA[under], self.tiles.tileDEC[under],
                    self.tiles.tileRA[over], self.tiles.tileDEC[over], tile_diameter)
            self.tile_over[passnum] = over

    def save(self):
        """Save a snapshot of our current state that can be restored.

        Snapshot is saved to $DESISURVEY_OUTPUT/plan_YYYYMMDD.fits using the
        date when :meth:`afternoon_plan` or :meth:`initialize` was last run.
        """
        if self.first_night is None:
            raise RuntimeError('Cannot save a plan before it has been initialized.')
        assert self.tile_priority is not None
        config = desisurvey.config.Configuration()
        name = config.get_path('plan_{}.fits'.format(self.last_night.isoformat()))
        t = astropy.table.Table(meta={
            'CADENCE': self.fiberassign_cadence,
            'FIRST': self.first_night.isoformat(),
            'LAST': self.last_night.isoformat() if self.last_night is not None else '',
            })
        t['COVERED'] = self.tile_covered
        t['COUNTDOWN'] = self.tile_countdown
        t['AVAILABLE'] = self.tile_available
        t['PRIORITY'] = self.tile_priority
        t.write(name, overwrite=True)

    def initialize(self, night):
        # Remember the first night of the survey.
        self.first_night = night
        self.last_night = night
        # Initialize priorities.
        if self.rules is not None:
            none_completed = np.zeros(self.tiles.ntiles, bool)
            self.tile_priority = self.rules.apply(none_completed)
            if not np.any(self.tile_priority > 0):
                raise RuntimeError('Initial tile priorities are all <= 0.')
        else:
            self.tile_priority = np.ones(self.tiles.ntiles, float)
        return self.tile_available, self.tile_priority

    def fiberassign(self, night, completed):
        # Calculate the number of elapsed nights in the survey.
        day_number = (night - self.first_night).days
        print('Running fiber assignment on {} (day number {}) with {} tiles completed.'
              .format(night, day_number, np.count_nonzero(completed)))
        for passnum in self.tiles.passes:
            under = self.tiles.passnum == passnum
            over = self.tile_over[passnum]
            if not np.any(over):
                continue
            overlapping = self.overlapping[passnum]
            # Identify all tiles in this pass whose covering tiles are completed.
            covered = np.all(~overlapping | completed[over], axis=1)
            # Which tiles have been newly covered since the last call to fiberassign?
            new_covered = covered & (self.tile_covered[under] == -1)
            if np.any(new_covered):
                new_tiles = self.tiles.tileID[under][new_covered]
               # Record the day number when these tiles were first covered.
                new = under.copy()
                new[under] = new_covered
                self.tile_covered[new] = day_number
        # Identify tiles that have been covered but not yet had fiber assignment run.
        ready = ~self.tile_available & (self.tile_covered >= 0)
        # Run fiber assignment on ready tiles that have completed their countdown.
        run_now = ready & (self.tile_countdown == 0)
        self.tile_available[run_now] = True
        # Update delay countdown for the remaining ready tiles.
        delayed = ready & (self.tile_countdown > 0)
        self.tile_countdown[delayed] -= 1
        self.log.info('Fiber assigned {} tiles, with {} delayed.'
                      .format(np.count_nonzero(run_now), np.count_nonzero(delayed)))

    def afternoon_plan(self, night, completed):
        if self.first_night is None:
            raise RuntimeError('Must call initialize() before afternoon_plan()')
        # Update fiber assignments this afternoon?
        if self.fiberassign_cadence == 'monthly':
            # Run fiber assignment on the afternoon before the full moon.
            dt = self.ephem.get_night(night)['nearest_full_moon']
            run_fiberassign = (dt > -0.5) and (dt <= 0.5)
            assert run_fiberassign == self.ephem.is_full_moon(night, num_nights=1)
        else:
            run_fiberassign = True
        if run_fiberassign:
            self.fiberassign(night, completed)
        # Update tile priorities.
        if self.rules is not None:
            self.tile_priority = self.rules.apply(completed)
        self.last_night = night
        return self.tile_available, self.tile_priority
