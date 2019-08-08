"""Utility functions for survey planning and scheduling.
"""
from __future__ import print_function, division

import datetime
import os
import warnings

import numpy as np

import pytz

import astropy.time
from astropy.coordinates import EarthLocation
import astropy.utils.iers
import astropy.utils.data
import astropy.utils.exceptions
import astropy._erfa.core
import astropy.units as u

import desiutil.log
import desiutil.iers

import desimodel.weather

from .config import Configuration



_telescope_location = None
#
# This global variable appears to be unused.
#
_dome_closed_fractions = None

# Temporary assignment for backward compatibility
freeze_iers = desiutil.iers.freeze_iers


def get_location():
    """Return the telescope's earth location.

    The location object is cached after the first call, so there is no need
    to cache this function's return value externally.

    Returns
    -------
    astropy.coordinates.EarthLocation
    """
    global _telescope_location
    if _telescope_location is None:
        config = Configuration()
        _telescope_location = EarthLocation.from_geodetic(
            lat=config.location.latitude(),
            lon=config.location.longitude(),
            height=config.location.elevation())
    return _telescope_location


def get_observer(when, alt=None, az=None):
    """Return the AltAz frame for the telescope at the specified time(s).

    Refraction corrections are not applied (for now).

    The returned object is automatically broadcast over input arrays.

    Parameters
    ----------
    when : astropy.time.Time
        One or more times when the AltAz transformations should be calculated.
    alt : astropy.units.Quantity or None
        Local altitude angle(s)
    az : astropy.units.Quantity or None
        Local azimuth angle(s)

    Returns
    -------
    astropy.coordinates.AltAz
        AltAz frame object suitable for transforming to/from local horizon
        (alt, az) coordinates.
    """
    if alt is not None and az is not None:
        kwargs = dict(alt=alt, az=az)
    elif alt is not None or az is not None:
        raise ValueError('Must specify both alt and az.')
    else:
        kwargs = {}
    return astropy.coordinates.AltAz(
        location=get_location(), obstime=when, pressure=0, **kwargs)


def cos_zenith_to_airmass(cosZ):
    """Convert a zenith angle to an airmass.

    Uses the Rozenberg 1966 interpolation formula, which gives reasonable
    results for high zenith angles, with a horizon air mass of 40.
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Interpolative_formulas
    Rozenberg, G. V. 1966. "Twilight: A Study in Atmospheric Optics."
    New York: Plenum Press, 160.

    The value of cosZ is clipped to [0,1], so observations below the horizon
    return the horizon value (~40).

    Parameters
    ----------
    cosZ : float or array
        Cosine of angle(s) to convert.

    Returns
    -------
    float or array
        Airmass value(s) >= 1.
    """
    cosZ = np.clip(np.asarray(cosZ), 0., 1.)
    return np.clip(1. / (cosZ + 0.025 * np.exp(-11 * cosZ)), 1., None)


def get_airmass(when, ra, dec):
    """Return the airmass of (ra,dec) at the specified observing time.

    Uses :func:`cos_zenith_to_airmass`.

    Parameters
    ----------
    when : astropy.time.Time
        Observation time, which specifies the local zenith.
    ra : astropy.units.Quantity
        Target RA angle(s)
    dec : astropy.units.Quantity
        Target DEC angle(s)
    Returns
    -------
    array or float
        Value of the airmass for each input (ra,dec).
    """
    target = astropy.coordinates.ICRS(ra=ra, dec=dec)
    zenith = get_observer(when, alt=90 * u.deg, az=0 * u.deg
                          ).transform_to(astropy.coordinates.ICRS)
    # Calculate zenith angle in degrees.
    zenith_angle = target.separation(zenith)
    # Convert to airmass.
    return cos_zenith_to_airmass(np.cos(zenith_angle))


def cos_zenith(ha, dec, latitude=None):
    """Calculate cos(zenith) for specified hour angle, DEC and latitude.

    Combine with :func:`cos_zenith_to_airmass` to calculate airmass.

    Parameters
    ----------
    ha : astropy.units.Quantity
        Hour angle(s) to use, with units convertible to angle.
    dec : astropy.units.Quantity
        Declination angle(s) to use, with units convertible to angle.
    latitude : astropy.units.Quantity or None
        Latitude angle to use, with units convertible to angle.
        Defaults to the latitude of :func:`get_location` if None.

    Returns
    -------
    numpy array
        cosine of zenith angle(s) corresponding to the inputs.
    """
    if latitude is None:
        # Use the observatory latitude by default.
        latitude = Configuration().location.latitude()
    # Calculate sin(altitude) = cos(zenith).
    cosZ = (np.sin(dec) * np.sin(latitude) +
            np.cos(dec) * np.cos(latitude) * np.cos(ha))
    # Return a plain array (instead of a unitless Quantity).
    return cosZ.value


def is_monsoon(night):
    """Test if this night's observing falls in the monsoon shutdown.

    Uses the monsoon date ranges defined in the
    :class:`desisurvey.config.Configuration`.

    Parameters
    ----------
    night : date
        Converted to a date using :func:`desisurvey.utils.get_date`.

    Returns
    -------
    bool
        True if this night's observing falls during the monsoon shutdown.
    """
    date = get_date(night)
    # Fetch our configuration.
    config = Configuration()
    # Test if date falls within any of the shutdowns.
    for key in config.monsoon.keys:
        node = getattr(config.monsoon, key)
        if date >= node.start() and date < node.stop():
            return True
    # If we get here, date does not fall in any of the shutdowns.
    return False


def local_noon_on_date(day):
    """Convert a date to an astropy time at local noon.

    Local noon is used as the separator between observing nights. The purpose
    of this function is to standardize the boundary between observing nights
    and the mapping of dates to times.

    Generates astropy ErfaWarnings for times in the future.

    Parameters
    ----------
    day : datetime.date
        The day to use for generating a time object.

    Returns
    -------
    astropy.time.Time
        A Time object with the input date and a time corresponding to
        local noon at the telescope.
    """
    # Fetch our configuration.
    config = Configuration()

    # Build a datetime object at local noon.
    tz = pytz.timezone(config.location.timezone())
    local_noon = tz.localize(
        datetime.datetime.combine(day, datetime.time(hour=12)))

    # Convert to UTC.
    utc_noon = local_noon.astimezone(pytz.utc)

    # Return a corresponding astropy Time.
    return astropy.time.Time(utc_noon)


def get_current_date():
    """Give current date following get_date convention (date changes at noon).

    Returns
    -------
    datetime.date object for current night, following get_date convention
    """
    date = datetime.datetime.now().astimezone()
    return get_date(date)


def get_date(date):
    """Convert different date specifications into a datetime.date object.

    We use strptime() to convert an input string, so leading zeros are not
    required for strings in the format YYYY-MM-DD, e.g. 2019-8-3 is considered
    valid.

    Instead of testing the input type, we try different conversion methods:
    ``.datetime.date()`` for an astropy time and ``datetime.date()`` for a
    datetime.

    Date specifications that include a time of day (datetime, astropy time, MJD)
    are rounded down to the previous local noon before converting to a date.
    This ensures that all times during a local observing night are mapped to
    the same date, when the night started.  A "naive" (un-localized) datetime
    is assumed to refer to UTC.

    Generates astropy ERFA warnings for future dates.

    Parameters
    ----------
    date : astropy.time.Time, datetime.date, datetime.datetime, string or number
        Specification of the date to return.  A string must have the format
        YYYY-MM-DD (but leading zeros on MM and DD are optional).  A number
        will be interpreted as a UTC MJD value.

    Returns
    -------
    datetime.date
    """
    input_date = date
    # valid types: string, number, Time, datetime, date
    try:
        # Convert bytes to str.
        date = date.decode()
    except AttributeError:
        pass
    try:
        # Convert a string of the form YYYY-MM-DD into a date.
        # This will raise a ValueError for a badly formatted string
        # or invalid date such as 2019-13-01.
        try:
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            try:
                date = datetime.datetime.strptime(date, '%Y%m%d').date()
            except ValueError:
                raise
    except TypeError:
        pass
    # valid types: number, Time, datetime, date
    try:
        # Convert a number to an astropy time, assuming it is a UTC MJD value.
        date = astropy.time.Time(date, format='mjd')
    except ValueError:
        pass
    # valid types: Time, datetime, date
    try:
        # Convert an astropy time into a datetime
        date = date.datetime
    except AttributeError:
        pass
    # valid types: datetime, date
    try:
        # Localize a naive datetime assuming it refers to UTC.
        date = pytz.utc.localize(date)
    except (AttributeError, ValueError):
        pass
    # valid types: localized datetime, date
    try:
        # Convert a localized datetime into the date of the previous noon.
        local_tz = pytz.timezone(
            Configuration().location.timezone())
        local_time = date.astimezone(local_tz)
        date = local_time.date()
        if local_time.hour < 12:
            date -= datetime.timedelta(days=1)
    except AttributeError:
        pass
    # valid types: date
    if not isinstance(date, datetime.date):
        raise ValueError('Invalid date specification: {0}.'.format(input_date))
    return date


def night_to_str(date):
    """Return DESI string format (YYYYMMDD) of datetime night.

    Parameters
    ----------
    date : datetime.date object, as from get_date()

    Returns
    -------
    str
        YYYMMDD formatted date string
    """
    return date.isoformat().replace('-', '')


def day_number(date):
    """Return the number of elapsed days since the start of the survey.

    Does not perform any range check that the date is within the nominal
    survey schedule.

    Parameters
    ----------
    date : astropy.time.Time, datetime.date, datetime.datetime, string or number
        Converted to a date using :func:`get_date`.

    Returns
    -------
    int
        Number of elapsed days since the start of the survey.
    """
    config = Configuration()
    return (get_date(date) - config.first_day()).days


def separation_matrix(ra1, dec1, ra2, dec2, max_separation=None):
    """Build a matrix of pair-wise separation between (ra,dec) pointings.

    The ra1 and dec1 arrays must have the same shape. The ra2 and dec2 arrays
    must also have the same shape, but it can be different from the (ra1,dec1)
    shape, resulting in a non-square return matrix.

    Uses the Haversine formula for better accuracy at low separations. See
    https://en.wikipedia.org/wiki/Haversine_formula for details.

    Equivalent to using the separations() method of astropy.coordinates.ICRS,
    but faster since it bypasses any units.

    Parameters
    ----------
    ra1 : array
        1D array of n1 RA coordinates in degrees (without units attached).
    dec1 : array
        1D array of n1 DEC coordinates in degrees (without units attached).
    ra2 : array
        1D array of n2 RA coordinates in degrees (without units attached).
    dec2 : array
        1D array of n2 DEC coordinates in degrees (without units attached).
    max_separation : float or None
        When present, the matrix elements are replaced with booleans given
        by (value <= max_separation), which saves some computation.

    Returns
    -------
    array
        Array with shape (n1,n2) with element [i1,i2] giving the 3D separation
        angle between (ra1[i1],dec1[i1]) and (ra2[i2],dec2[i2]) in degrees
        or, if max_separation is not None, booleans (value <= max_separation).
    """
    ra1, ra2 = np.deg2rad(ra1), np.deg2rad(ra2)
    dec1, dec2 = np.deg2rad(dec1), np.deg2rad(dec2)
    if ra1.shape != dec1.shape:
        raise ValueError('Arrays ra1, dec1 must have the same shape.')
    if len(ra1.shape) != 1:
        raise ValueError('Arrays ra1, dec1 must be 1D.')
    if ra2.shape != dec2.shape:
        raise ValueError('Arrays ra2, dec2 must have the same shape.')
    if len(ra2.shape) != 1:
        raise ValueError('Arrays ra2, dec2 must be 1D.')
    havRA12 = 0.5 * (1 - np.cos(ra2 - ra1[:, np.newaxis]))
    havDEC12 = 0.5 * (1 - np.cos(dec2 - dec1[:, np.newaxis]))
    havPHI = havDEC12 + np.cos(dec1)[:, np.newaxis] * np.cos(dec2) * havRA12
    if max_separation is not None:
        # Replace n1 x n2 arccos calls with a single sin call.
        threshold = np.sin(0.5 * np.deg2rad(max_separation)) ** 2
        return havPHI <= threshold
    else:
        return np.rad2deg(np.arccos(np.clip(1 - 2 * havPHI, -1, +1)))


def lb2uv(l, b):
    """Convert longitude and latitude to unit vectors on sphere.

    Parameters
    ----------
    l : array
      1D array of N RA coordinates in degrees (without units)
    b : array
      1D array of N DEC coordinates in degrees (without units)

    Returns
    -------
    array
        Array with shape [N, 3] giving unit vectors corresponding to l, b.
    """
    t = np.deg2rad(90-b)
    p = np.deg2rad(l)
    z = np.cos(t)
    x = np.cos(p)*np.sin(t)
    y = np.sin(p)*np.sin(t)
    return np.concatenate([q[..., np.newaxis] for q in (x, y, z)],
                          axis=-1)


def gc_dist(lon1, lat1, lon2, lat2):
    """Compute distance on sphere in degrees between lon1, lat1 and lon2, lat2.
    """

    from numpy import sin, cos, arcsin, sqrt

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    return np.degrees(
        2*arcsin(sqrt((sin((lat1-lat2)*0.5))**2 +
                      cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2)))


def match_radec(r1, d1, r2, d2, rad=1./60./60., nneighbor=0, notself=False):
    """Return matches between points (r1, d1) and (r2, d2) within radius.

    Uses kd-trees to avoid making all N^2 comparisons; much faster than
    separation_matrix when there are few matches relative to N^2.

    Parameters
    ----------
    r1 : array
        1D array of n1 RA coordinates in degrees (without units attached)
    d1 : array
        1D array of n1 DEC coordinates in degrees (without units attached)
    r2 : array
        1D array of n2 RA coordinates in degrees (without units attached)
    d2 : array
        1D array of n2 DEC coordinates in degrees (without units attached)
    rad : float
        find matches out to rad degrees
    nneighbor : int
        find up to nneighbor matches
    notself : boolean
        if the r1 and d1, r2 and and d2 arrays are identical, do not report
        self matches.

    Returns
    -------
    m1, m2, d12
    m1 : array (int)
       indices into (r1, d1) of matches
    m2 : array (int)
       indices into (r2, d2) of matches
    d12 : array (float)
       separations between each match
    """
    # warning: cKDTree has issues if there are large numbers of points
    # at the exact same positions (it takes forever / reaches maximum
    # recursion depth).
    if notself and nneighbor > 0:
        nneighbor += 1
    uv1 = lb2uv(r1, d1)
    uv2 = lb2uv(r2, d2)
    from scipy.spatial import cKDTree
    tree = cKDTree(uv2)
    dub = 2*np.sin(np.radians(rad)/2)
    if nneighbor > 0:
        d12, m2 = tree.query(uv1, nneighbor, distance_upper_bound=dub)
        if nneighbor > 1:
            m2 = m2.reshape(-1)
            d12 = d12.reshape(-1)

        m1 = np.arange(len(r1)*nneighbor, dtype='i4') // nneighbor
        d12 = 2*np.arcsin(np.clip(d12 / 2, 0, 1))*180/np.pi
        m = m2 < len(r2)
    else:
        tree1 = cKDTree(uv1)
        res = tree.query_ball_tree(tree1, dub)
        lens = [len(r) for r in res]
        m2 = np.repeat(np.arange(len(r2), dtype='i4'), lens)
        if len(m2) > 0:
            m1 = np.concatenate([r for r in res if len(r) > 0])
        else:
            m1 = m2.copy()
        d12 = gc_dist(r1[m1], d1[m1], r2[m2], d2[m2])
        m = np.ones(len(m1), dtype='bool')
    if notself:
        m = m & (m1 != m2)
    return m1[m], m2[m], d12[m]
