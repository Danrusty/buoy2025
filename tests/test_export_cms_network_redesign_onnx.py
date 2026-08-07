"""CMS 网络重构 ONNX 冻结脚本的安全约束测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from export_cms_network_redesign_onnx import (  # noqa: E402
    COMPATIBILITY_ALIAS,
    ONNX_FILENAME,
    RELEASE_DIR,
    RELEASE_VERSION,
    WINDOWS_STAGING_PATH,
)


class ExportCmsNetworkRedesignOnnxTest(unittest.TestCase):
    def test_release_is_isolated_but_keeps_requested_onnx_filename(self) -> None:
        self.assertEqual(
            RELEASE_DIR.name,
            "wdf_cms_network_redesign_v1",
        )
        self.assertEqual(RELEASE_VERSION, RELEASE_DIR.name)
        self.assertEqual(
            ONNX_FILENAME,
            "wdf_cms_orig_core6_v1.onnx",
        )
        self.assertEqual(COMPATIBILITY_ALIAS, "wdf_drifter.onnx")
        self.assertTrue(
            WINDOWS_STAGING_PATH.endswith(
                r"\wdf_cms_network_redesign_v1"
            )
        )


if __name__ == "__main__":
    unittest.main()
