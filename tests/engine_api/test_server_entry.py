"""``python -m engine.server`` must use the same event classes as ``engine.server`` (which backends import)."""

import runpy
import sys
from unittest import mock


def test_main_entry_runs_the_imported_module():
    import engine.server as srv

    with mock.patch.object(srv, "main", return_value=0) as m, mock.patch.object(sys, "argv", ["engine.server"]):
        try:
            runpy.run_module("engine.server", run_name="__main__")
        except SystemExit as e:
            assert e.code == 0
    m.assert_called_once()
