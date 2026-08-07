import unittest

import numpy as np
from instatarget.core.errors import DepthError, ProtocolError
from instatarget.core.types import (
    BBoxXYWH,
    DepthPlane,
    FrameIndex,
    FramePacket,
    SequenceId,
    SphericalPoint,
)


class CoreTypesTest(unittest.TestCase):
    def testFramePacketAcceptsAlignedRgbAndDepth(self) -> None:
        rgb = np.zeros((4, 8, 3), dtype=np.uint8)
        depth = DepthPlane(
            values=np.ones((4, 8), dtype=np.float32),
            validMask=np.ones((4, 8), dtype=np.bool_),
            unit="m",
        )

        packet = FramePacket(
            sequenceId=SequenceId("sequence"),
            frameIndex=FrameIndex(0),
            timestampNs=0,
            rgb=rgb,
            depth=depth,
        )

        self.assertIs(packet.depth, depth)

    def testFramePacketRejectsMisalignedDepth(self) -> None:
        rgb = np.zeros((4, 8, 3), dtype=np.uint8)
        depth = DepthPlane(
            values=np.ones((2, 8), dtype=np.float32),
            validMask=np.ones((2, 8), dtype=np.bool_),
            unit="m",
        )

        with self.assertRaises(ProtocolError):
            FramePacket(
                sequenceId=SequenceId("sequence"),
                frameIndex=FrameIndex(0),
                timestampNs=0,
                rgb=rgb,
                depth=depth,
            )

    def testDepthPlaneRejectsWrongDtype(self) -> None:
        with self.assertRaises(DepthError):
            DepthPlane(
                values=np.ones((2, 2), dtype=np.float64),
                validMask=np.ones((2, 2), dtype=np.bool_),
                unit="m",
            )

    def testDepthPlaneRejectsNegativeValidDepth(self) -> None:
        with self.assertRaises(DepthError):
            DepthPlane(
                values=np.full((2, 2), -1.0, dtype=np.float32),
                validMask=np.ones((2, 2), dtype=np.bool_),
                unit="m",
            )

    def testSphericalPointRejectsNonUnitVector(self) -> None:
        with self.assertRaises(ProtocolError):
            SphericalPoint(x=2.0, y=0.0, z=0.0, yawRad=0.0, pitchRad=0.0)

    def testBoundingBoxRejectsNonPositiveDimensions(self) -> None:
        with self.assertRaises(ProtocolError):
            BBoxXYWH(xPx=0.0, yPx=0.0, widthPx=0.0, heightPx=1.0)


if __name__ == "__main__":
    unittest.main()
