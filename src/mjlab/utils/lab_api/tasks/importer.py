# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utility for importing all modules in a package recursively."""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys

logger = logging.getLogger(__name__)


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
  """Import all sub-packages in a package recursively.

  It is easier to use this function to import all sub-packages in a package recursively
  than to manually import each sub-package.

  It replaces the need of the following code snippet on the top of each package's ``__init__.py`` file:

  .. code-block:: python

      import .locomotion.velocity
      import .manipulation.reach
      import .manipulation.lift

  A sub-package that fails to import (e.g. an optional/demo task config whose
  heavy or stripped dependency is not installed — such as ``mjlab.asset_zoo``
  robot assets, which are dropped from the slim cyberwave_rl fat wheel) is
  logged and skipped rather than aborting the whole walk. This keeps one
  optional package from taking down discovery of every other task.

  Args:
      package_name: The package name.
      blacklist_pkgs: The list of blacklisted packages to skip. Defaults to None,
          which means no packages are blacklisted.
  """
  # Default blacklist
  if blacklist_pkgs is None:
    blacklist_pkgs = []

  def _on_import_error(failed_name: str) -> None:
    # Called from within the walk's `except` block, so sys.exc_info() still
    # holds the active exception. Log (don't swallow silently) so a skipped
    # package is diagnosable in stderr, then continue the walk.
    logger.warning(
      "import_packages: skipping %r — failed to import during recursive walk: %s",
      failed_name,
      sys.exc_info()[1],
    )

  # Import the package itself
  package = importlib.import_module(package_name)
  # Import all Python files
  for _ in _walk_packages(
    package.__path__,
    package.__name__ + ".",
    onerror=_on_import_error,
    blacklist_pkgs=blacklist_pkgs,
  ):
    pass


def _walk_packages(
  path: str | None = None,
  prefix: str = "",
  onerror: callable | None = None,
  blacklist_pkgs: list[str] | None = None,
):
  """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

  Note:
      This function is a modified version of the original ``pkgutil.walk_packages`` function. It adds
      the `blacklist_pkgs` argument to skip blacklisted packages. Please refer to the original
      ``pkgutil.walk_packages`` function for more details.
  """
  if blacklist_pkgs is None:
    blacklist_pkgs = []

  def seen(p, m={}):
    if p in m:
      return True
    m[p] = True  # noqa: R503

  for info in pkgutil.iter_modules(path, prefix):
    # check blacklisted
    if any([black_pkg_name in info.name for black_pkg_name in blacklist_pkgs]):
      continue

    # yield the module info
    yield info

    if info.ispkg:
      try:
        __import__(info.name)
      except Exception:
        if onerror is not None:
          onerror(info.name)
        else:
          raise
      else:
        path = getattr(sys.modules[info.name], "__path__", None) or []

        # don't traverse path items we've seen before
        path = [p for p in path if not seen(p)]

        yield from _walk_packages(path, info.name + ".", onerror, blacklist_pkgs)
