from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from pipeline_logging import configure_console, pipeline_print, stage_complete


class PipelineLoggingTests(unittest.TestCase):
    def tearDown(self):
        configure_console(False)

    def test_info_suppresses_detail_but_keeps_warning_failure_and_stage_progress(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.dict(os.environ, {"LOG_LEVEL": "INFO"}):
            configure_console(False)
            pipeline_print("processing slice 42")
            pipeline_print("WARNING: heart mask needs review")
            pipeline_print("Stage failed: missing checkpoint")
            stage_complete(2, "02")
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "WARNING: heart mask needs review",
                "Stage failed: missing checkpoint",
                "[02/07] Stage 02 Complete",
            ],
        )

    def test_verbose_and_debug_environment_reveal_details(self):
        for configure, environment in (
            (True, "INFO"),
            (False, "DEBUG"),
        ):
            output = io.StringIO()
            with redirect_stdout(output), patch.dict(os.environ, {"LOG_LEVEL": environment}):
                configure_console(configure)
                pipeline_print("processing slice 42")
            self.assertEqual(output.getvalue(), "processing slice 42\n")

    def test_stage_number_is_validated(self):
        with self.assertRaises(ValueError):
            stage_complete(8, "08")


if __name__ == "__main__":
    unittest.main()
