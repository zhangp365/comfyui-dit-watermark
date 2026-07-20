from __future__ import annotations

import unittest

from grow_dit.ecc import MAX_MESSAGE_BYTES, decode_frame, encode_frame, rs_encode


class ReedSolomonFrameTests(unittest.TestCase):
    def test_frame_round_trip(self) -> None:
        encoded = encode_frame("watermark")
        self.assertEqual(len(encoded), 14)
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
        for first, second in ((1, 2), (4, 9), (0, 13), (10, 12)):
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
        encoded[12] ^= 0x44
        self.assertFalse(decode_frame(bytes(encoded)).valid)

    def test_payload_limit_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, f"maximum is {MAX_MESSAGE_BYTES}"):
            encode_frame("x" * (MAX_MESSAGE_BYTES + 1))

    def test_frame_length_tracks_utf8_payload(self) -> None:
        self.assertEqual(len(encode_frame("a")), 6)
        self.assertEqual(len(encode_frame("水印")), 11)

    def test_legacy_zero_padded_frame_still_decodes(self) -> None:
        payload = b"watermark"
        legacy_data = bytes([len(payload)]) + payload + bytes(28 - 1 - len(payload))
        decoded = decode_frame(rs_encode(legacy_data))
        self.assertTrue(decoded.valid)
        self.assertEqual(decoded.message, "watermark")


if __name__ == "__main__":
    unittest.main()
