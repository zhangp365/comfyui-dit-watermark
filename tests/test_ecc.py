from __future__ import annotations

import unittest

from grow_dit.ecc import FRAME_BYTES, decode_frame, encode_frame


class ReedSolomonFrameTests(unittest.TestCase):
    def test_frame_round_trip(self) -> None:
        encoded = encode_frame("watermark")
        self.assertEqual(len(encoded), FRAME_BYTES)
        decoded = decode_frame(encoded)
        self.assertTrue(decoded.valid)
        self.assertEqual(decoded.message, "watermark")
        self.assertEqual(decoded.corrected_symbols, 0)

    def test_one_symbol_error_is_corrected(self) -> None:
        encoded = bytearray(encode_frame("watermark"))
        encoded[4] ^= 0xA7
        decoded = decode_frame(bytes(encoded))
        self.assertTrue(decoded.valid)
        self.assertEqual(decoded.message, "watermark")
        self.assertEqual(decoded.corrected_symbols, 1)

    def test_two_character_symbol_errors_are_corrected(self) -> None:
        for first, second in ((1, 2), (4, 9), (0, 31), (27, 30)):
            with self.subTest(first=first, second=second):
                encoded = bytearray(encode_frame("watermark"))
                encoded[first] ^= 0x53
                encoded[second] ^= 0xCA
                decoded = decode_frame(bytes(encoded))
                self.assertTrue(decoded.valid)
                self.assertEqual(decoded.message, "watermark")
                self.assertEqual(decoded.corrected_symbols, 2)

    def test_three_symbol_errors_are_rejected(self) -> None:
        encoded = bytearray(encode_frame("watermark"))
        encoded[1] ^= 0x11
        encoded[8] ^= 0x22
        encoded[20] ^= 0x44
        self.assertFalse(decode_frame(bytes(encoded)).valid)

    def test_payload_limit_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum is 27"):
            encode_frame("x" * 28)


if __name__ == "__main__":
    unittest.main()
