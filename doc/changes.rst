=====================
desisurvey change log
=====================

0.15.1 (unreleased)
-------------------

* No changes yet

0.15.0 (2021-02-15)
-------------------

* NTS updates for SV tiles (PRs `#123`_ and `#124`).

.. _`#123`: https://github.com/desihub/desisurvey/pull/123
.. _`#124`: https://github.com/desihub/desisurvey/pull/124

0.14.1 (2020-12-11)
-------------------

* Update desisurvey.plots.plot_sky_passes for compatibility with
  desiutil >= 3.0.0 (PR `#122`_).

.. _`#122`: https://github.com/desihub/desisurvey/pull/122

0.14.0 (2020-08-03)
-------------------

* Fix py3.8 invalid escape sequences (PR `#120`_).
* Update rules.yaml to use new tile file pass ordering (PR `#114`_);
  requires desimodel>=0.11.0
* Move IERS functions to desiutil_ (PR `#113`_).

.. _`#120`: https://github.com/desihub/desisurvey/pull/120
.. _`#114`: https://github.com/desihub/desisurvey/pull/114
.. _desiutil: https://github.com/desihub/desiutil
.. _`#113`: https://github.com/desihub/desisurvey/pull/113

0.13.0 (2020-04-07)
-------------------

Requires desimodel/0.12.0 or later with new tile file.

* Change fiber_assignment_order to reflect new pass ordering (PR `#111`_).
* Enable using NTS/desisurvey with CMX tile file; add HA limit (PR `#107`_).

.. _`#107`: https://github.com/desihub/desisurvey/pull/107
.. _`#111`: https://github.com/desihub/desisurvey/pull/111

0.12.1 (2019-12-20)
-------------------

* Workaround for missing IERS server (PR `#105`_).

.. _`#105`: https://github.com/desihub/desisurvey/pull/105

0.12.0 (2019-08-09)
-------------------

* Minor updates to conform to data model standards (PR `#94`_).
* Improved documentation (PR `#94`_).
* Increase tile radius for coverage check (PR `#97`_).
* Fix RA,DEC vs. DEC,RA bug (PR `#99`_).
* Adds `desisurvey.scheduler.NTS` (Next Tile Selector) interface to ICS
  (PR `#99`_)

.. _`#94`: https://github.com/desihub/desisurvey/pull/94
.. _`#97`: https://github.com/desihub/desisurvey/pull/97
.. _`#99`: https://github.com/desihub/desisurvey/pull/99

0.11.0 (2018-11-26)
-------------------

This version is a major refactoring of the code to simplify the logic
for easier maintenance and documentation. There is now a clean
separation between survey strategy, afternoon planning,
next-tile selection, and exposure-time calculations. The refactored
code is also significantly faster (PR `#91`_).

* Add new modules: tiles, forecast, scheduler.
* Move modules schedule, progress, surveyplan to old/.
* Add new class ExposureTimeCalculator to etc module.
* Add new class Planner to plan module.
* Decouple ephemerides date range from nominal survey start/stop.
* Rename ephemerides to ephem (to enforce new get_ephem access pattern).
* Use of twilight is now optional and off by default.
* Exposure times include an average correction for the moon: this will
  be fixed in a future release.

.. _`#91`: https://github.com/desihub/desisurvey/pull/91

0.10.4 (2018-10-02)
-------------------

Updates for survey margin estimates (PR `#89`_):

* Implement realistic 18-day monsoon shutdowns instead of fixed 45-day period.
* Replay daily Mayall weather history instead of fixed monthly fractions
  (needs desimodel >= 0.9.8)
* Update exposure-time model for atmospheric seeing.
* Speed up full-moon, program change and LST calculations in ephemerides module.
* Requires desimodel >= 0.9.8

.. _`#89`: https://github.com/desihub/desisurvey/pull/89

0.10.3 (2018-09-26)
-------------------

* Added tiling dithering and QA code (PR `#87`_).
* Allow ``PASS`` to be as large as 99 (PR `#88`_).

.. _`#87`: https://github.com/desihub/desisurvey/pull/87
.. _`#88`: https://github.com/desihub/desisurvey/pull/88

0.10.2 (2018-06-27)
-------------------

* Do not assume that input tile file includes all of DARK, BRIGHT, and GRAY
  tiles (PR `#83`_).
* Enforce at least six characters in program name in exposures table (PR `#86`_).

.. _`#83`: https://github.com/desihub/desisurvey/pull/83
.. _`#86`: https://github.com/desihub/desisurvey/pull/86

0.10.1 (2017-12-20)
-------------------

* Set the ``EXTNAME`` keyword on the Table returned by ``Progress.get_exposures()``.

0.10.0 (2017-11-09)
-------------------

* Progress.get_exposures() updates:

  * includes FLAVOR and PROGRAM columns.
  * uses desimodel.footprint.pass2program if available.
  * standardized on UPPERCASE column names and NIGHT=YEARMMDD not YEAR-MM-DD.

0.9.3 (2017-10-09)
------------------

* Fixes #18, #49, #54.
* Improvements to surveymovie script.
* Add progress columns to track fiber assignment and planning.
* Add support for optional depth-first survey strategy.
* Docs now auto-generated at http://desisurvey.readthedocs.io/en/latest/

0.9.2 (2017-09-29)
------------------

* Implement fiber assignment policy via --fa-delay option to surveyplan.

0.9.1 (2017-09-20)
------------------

* Command line scripts --config-file option to override default config file.
* Fixes for bugs that occur when testing with a small subset of tiles.
* Changes $DESISURVEY -> $DESISURVEY_OUTPUT as output dir envvar name
* Remove astropy units from function signatures (for readthedocs).
* Add travis, coveralls and readthedocs automation.

0.9.0 (2017-09-11)
------------------

* Create surveyinit script to calculate initial HA assignments.
* Improve Optimizer algorithms (~10x faster, better initialization).
* Create surveymovie to visualize survey scheduling and progress.
* Rework surveyplan to track fiber assignment availability.
* Validate a set of observing rules consistent with the baseline strategy
  described in DESI-doc-1767-v3.

0.8.2 (2017-07-12)
------------------

* Fix flat vs. flatten for older versions of numpy (PR `#52`_).

.. _`#52`: https://github.com/desihub/desisurvey/pull/52

0.8.1 (2017-06-19)
------------------

* Fix unit tests broken in 0.8.0 (PR `#46`_).

.. _`#46`: https://github.com/desihub/desisurvey/pull/46

0.8.0 (2017-06-18)
------------------

* Implement LST-driven scheduling strategy.
* Create new optimize module for iterative HA optimization.
* Rename module plan -> schedule.
* Create new plan module to manage fiber-assignment groups and priorities.

0.7.0 (2017-06-05)
------------------

* Freeze IERS table used by astropy time, coordinates.
* Implement alternate greedy scheduler with optional policy weights.
* Add `plots.plot_scheduler()`
* Partial fix of RA=0/360 planning bug

0.6.0 (2017-05-10)
------------------

* Add new config yaml file and python wrapper.
* Convert all code to use new config machinery.
* Add new class Plan for future use in scheduling.
* Unify different output files with overlapping contents into single output
  managed by desisurvey.progress.
* Cleanup and reorganize the Ephemerides class.
* Add comparisons with independent JPL Horizons run to unit tests for
  AltAz transforms and ephemerides calculations.
* Add new plot utilities for Progress and Plan objects.
* Document and handle astropy IERS warnings about future times.
* Rename exposurecalc module to etc (exposure-time calculator).
* Update docstrings and imports, and remove unused code.

0.5.0 (2017-04-13)
------------------

* Add new plot methods
* Bug fix to Az computation and airmass calculator
* Code reorganization

0.4.0 (2017-04-04)
------------------

This version was tagged for the 2% sprint data challenge.

* Add unit tests; fix afternoon planning tile updates and other minor bugs
* Fix off-by-one with YEARMMDD vs. MJD of sunset
* Add new plots module
* Refactor nightcal module into ephmerides

0.3.1 (2016-12-21)
------------------

* fixed E(B-V) scaling for exposure time (PR #12)

0.3.0 (2016-11-29)
------------------

First release after refactoring.

0.2.0 (2016-11-19)
------------------

Last version before repackaging of surveysim.
