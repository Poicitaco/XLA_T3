"""
Unit Tests for Face Lock/Unlock Roundtrip
Tests encryption reversibility, data integrity, and edge cases.
"""

import base64
import unittest

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.face_lock import EncryptedFaceRegion, FaceRegionLocker


class TestFaceRegionLocker(unittest.TestCase):
    """Test suite for FaceRegionLocker encryption/decryption."""

    def setUp(self):
        """Set up test fixtures."""
        self.passphrase = "test_passphrase_2026"
        self.locker = FaceRegionLocker(passphrase=self.passphrase)
        
        # Create a synthetic test frame
        self.test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Common test boxes
        self.single_box = [[100, 100, 80, 100]]
        self.multiple_boxes = [
            [50, 50, 60, 75],
            [200, 150, 70, 90],
            [400, 300, 85, 110],
        ]

    def test_empty_passphrase_raises_error(self):
        """Test that empty passphrase raises ValueError."""
        with self.assertRaises(ValueError):
            FaceRegionLocker(passphrase="")

    def test_key_derivation_deterministic(self):
        """Test that same passphrase generates same key."""
        locker1 = FaceRegionLocker(passphrase="test123")
        locker2 = FaceRegionLocker(passphrase="test123")
        
        self.assertEqual(locker1._key, locker2._key)

    def test_different_passphrase_different_key(self):
        """Test that different passphrases generate different keys."""
        locker1 = FaceRegionLocker(passphrase="test123")
        locker2 = FaceRegionLocker(passphrase="test456")
        
        self.assertNotEqual(locker1._key, locker2._key)

    def test_lock_unlock_roundtrip_single_face(self):
        """Test perfect reconstruction with single face."""
        locked_frame, payloads = self.locker.lock_faces(
            self.test_frame,
            self.single_box,
            overlay_mode="rps",
        )
        
        self.assertEqual(len(payloads), 1)
        self.assertFalse(np.array_equal(locked_frame, self.test_frame))
        
        # Unlock
        unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)
        
        # Should be identical to original
        np.testing.assert_array_equal(unlocked_frame, self.test_frame)

    def test_lock_unlock_roundtrip_multiple_faces(self):
        """Test perfect reconstruction with multiple faces."""
        locked_frame, payloads = self.locker.lock_faces(
            self.test_frame,
            self.multiple_boxes,
            overlay_mode="ciphernoise",
        )
        
        self.assertEqual(len(payloads), 3)
        
        unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)
        
        np.testing.assert_array_equal(unlocked_frame, self.test_frame)

    def test_all_overlay_modes_roundtrip(self):
        """Test that all overlay modes maintain perfect roundtrip."""
        modes = ["solid", "noise", "ciphernoise", "rps"]
        
        for mode in modes:
            with self.subTest(mode=mode):
                locked_frame, payloads = self.locker.lock_faces(
                    self.test_frame,
                    self.single_box,
                    overlay_mode=mode,
                )
                
                unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)
                
                np.testing.assert_array_equal(
                    unlocked_frame,
                    self.test_frame,
                    err_msg=f"Roundtrip failed for mode: {mode}",
                )

    def test_empty_boxes_returns_original_frame(self):
        """Test that empty boxes list returns unchanged frame."""
        locked_frame, payloads = self.locker.lock_faces(self.test_frame, [])
        
        np.testing.assert_array_equal(locked_frame, self.test_frame)
        self.assertEqual(len(payloads), 0)

    def test_empty_frame_handled_gracefully(self):
        """Test that empty frame is handled without crashing."""
        empty_frame = np.array([], dtype=np.uint8)
        
        locked_frame, payloads = self.locker.lock_faces(empty_frame, self.single_box)
        
        self.assertEqual(locked_frame.size, 0)
        self.assertEqual(len(payloads), 0)

    def test_unlock_with_empty_payloads(self):
        """Test unlock with no payloads returns frame unchanged."""
        unlocked = self.locker.unlock_faces(self.test_frame, [])
        
        np.testing.assert_array_equal(unlocked, self.test_frame)

    def test_wrong_passphrase_fails_decryption(self):
        """Test that wrong passphrase fails to decrypt."""
        locked_frame, payloads = self.locker.lock_faces(
            self.test_frame,
            self.single_box,
        )
        
        # Try to unlock with different passphrase
        wrong_locker = FaceRegionLocker(passphrase="wrong_passphrase")
        
        with self.assertRaises(Exception):  # Should raise cryptography exception
            wrong_locker.unlock_faces(locked_frame, payloads)

    def test_payload_serialization_roundtrip(self):
        """Test JSON serialization and deserialization of payloads."""
        _, payloads = self.locker.lock_faces(
            self.test_frame,
            self.multiple_boxes,
        )
        
        # Serialize to JSON
        json_str = FaceRegionLocker.dumps_payloads(payloads)
        self.assertIsInstance(json_str, str)
        
        # Deserialize
        jsonable = FaceRegionLocker.payloads_to_jsonable(payloads)
        restored_payloads = FaceRegionLocker.payloads_from_jsonable(jsonable)
        
        # Check equality
        self.assertEqual(len(restored_payloads), len(payloads))
        
        for orig, restored in zip(payloads, restored_payloads):
            self.assertEqual(orig.box, restored.box)
            self.assertEqual(orig.nonce_b64, restored.nonce_b64)
            self.assertEqual(orig.ciphertext_b64, restored.ciphertext_b64)
            self.assertEqual(orig.shape, restored.shape)

    def test_box_out_of_bounds_handled(self):
        """Test that out-of-bounds boxes are handled gracefully."""
        out_of_bounds_box = [[10000, 10000, 50, 50]]
        
        locked_frame, payloads = self.locker.lock_faces(
            self.test_frame,
            out_of_bounds_box,
        )
        
        # Should not crash, may not create payload
        self.assertIsInstance(locked_frame, np.ndarray)

    def test_tiny_box_handled(self):
        """Test that very small boxes are handled."""
        tiny_box = [[100, 100, 1, 1]]
        
        locked_frame, payloads = self.locker.lock_faces(
            self.test_frame,
            tiny_box,
        )
        
        # Should skip tiny boxes
        self.assertEqual(len(payloads), 0)

    def test_head_expansion_ratios(self):
        """Test different head expansion ratios."""
        ratios = [0.0, 0.5, 0.7, 1.0, 2.0]
        
        for ratio in ratios:
            with self.subTest(ratio=ratio):
                locked_frame, payloads = self.locker.lock_faces(
                    self.test_frame,
                    self.single_box,
                    head_ratio=ratio,
                )
                
                unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)
                
                np.testing.assert_array_equal(unlocked_frame, self.test_frame)

    def test_rps_parameters_roundtrip(self):
        """Test different RPS tile sizes and rounds."""
        params = [
            {"rps_tile_size": 4, "rps_rounds": 1},
            {"rps_tile_size": 8, "rps_rounds": 2},
            {"rps_tile_size": 16, "rps_rounds": 3},
        ]
        
        for param in params:
            with self.subTest(param=param):
                locked_frame, payloads = self.locker.lock_faces(
                    self.test_frame,
                    self.single_box,
                    overlay_mode="rps",
                    **param,
                )
                
                unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)
                
                np.testing.assert_array_equal(unlocked_frame, self.test_frame)

    def test_nonce_uniqueness(self):
        """Test that each encryption uses a unique nonce."""
        _, payloads1 = self.locker.lock_faces(self.test_frame, self.single_box)
        _, payloads2 = self.locker.lock_faces(self.test_frame, self.single_box)
        
        nonce1 = payloads1[0].nonce_b64
        nonce2 = payloads2[0].nonce_b64
        
        self.assertNotEqual(nonce1, nonce2)

    def test_ciphertext_differs_across_encryptions(self):
        """Test that same data produces different ciphertext (due to nonce)."""
        _, payloads1 = self.locker.lock_faces(self.test_frame, self.single_box)
        _, payloads2 = self.locker.lock_faces(self.test_frame, self.single_box)
        
        ct1 = payloads1[0].ciphertext_b64
        ct2 = payloads2[0].ciphertext_b64
        
        self.assertNotEqual(ct1, ct2)

    def test_payload_integrity_check(self):
        """Test that tampering with payload fails decryption."""
        locked_frame, payloads = self.locker.lock_faces(
            self.test_frame,
            self.single_box,
        )
        
        # Tamper with ciphertext
        tampered_payload = EncryptedFaceRegion(
            box=payloads[0].box,
            nonce_b64=payloads[0].nonce_b64,
            ciphertext_b64=base64.b64encode(b"tampered_data").decode(),
            shape=payloads[0].shape,
        )
        
        with self.assertRaises(Exception):
            self.locker.unlock_faces(locked_frame, [tampered_payload])

    def test_different_frame_sizes(self):
        """Test with different frame resolutions."""
        sizes = [
            (240, 320, 3),
            (480, 640, 3),
            (720, 1280, 3),
            (1080, 1920, 3),
        ]
        
        for size in sizes:
            with self.subTest(size=size):
                frame = np.random.randint(0, 256, size, dtype=np.uint8)
                box = [[50, 50, 40, 50]]
                
                locked_frame, payloads = self.locker.lock_faces(frame, box)
                unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)
                
                np.testing.assert_array_equal(unlocked_frame, frame)

    def test_real_image_roundtrip(self):
        """Test with a real image pattern."""
        # Create a more realistic test image with gradients
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add gradient
        for i in range(h):
            frame[i, :] = (i * 255 // h, i * 128 // h, 128)
        
        locked_frame, payloads = self.locker.lock_faces(
            frame,
            self.single_box,
        )
        
        unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)
        
        np.testing.assert_array_equal(unlocked_frame, frame)

    def test_edge_boxes_near_boundary(self):
        """Test boxes at frame edges."""
        h, w = self.test_frame.shape[:2]
        
        edge_boxes = [
            [0, 0, 50, 50],  # Top-left
            [w - 50, 0, 50, 50],  # Top-right
            [0, h - 50, 50, 50],  # Bottom-left
            [w - 50, h - 50, 50, 50],  # Bottom-right
        ]
        
        locked_frame, payloads = self.locker.lock_faces(
            self.test_frame,
            edge_boxes,
        )
        
        unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)
        
        np.testing.assert_array_equal(unlocked_frame, self.test_frame)


class TestEncryptedFaceRegion(unittest.TestCase):
    """Test EncryptedFaceRegion dataclass."""

    def test_creation(self):
        """Test creating EncryptedFaceRegion."""
        region = EncryptedFaceRegion(
            box=[10, 20, 30, 40],
            nonce_b64="test_nonce",
            ciphertext_b64="test_cipher",
            shape=[50, 60, 3],
        )
        
        self.assertEqual(region.box, [10, 20, 30, 40])
        self.assertEqual(region.nonce_b64, "test_nonce")
        self.assertEqual(region.ciphertext_b64, "test_cipher")
        self.assertEqual(region.shape, [50, 60, 3])


class TestPixelShuffling(unittest.TestCase):
    """Test reversible pixel shuffling algorithm."""

    def test_reversible_pixel_shuffle_deterministic(self):
        """Test that same seed produces same shuffle."""
        roi = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        seed = 12345
        
        shuffled1 = FaceRegionLocker._reversible_pixel_shuffle(roi, seed, tile_size=8, rounds=2)
        shuffled2 = FaceRegionLocker._reversible_pixel_shuffle(roi, seed, tile_size=8, rounds=2)
        
        np.testing.assert_array_equal(shuffled1, shuffled2)

    def test_different_seed_different_shuffle(self):
        """Test that different seeds produce different results."""
        roi = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        shuffled1 = FaceRegionLocker._reversible_pixel_shuffle(roi, seed=111, tile_size=8, rounds=2)
        shuffled2 = FaceRegionLocker._reversible_pixel_shuffle(roi, seed=222, tile_size=8, rounds=2)
        
        self.assertFalse(np.array_equal(shuffled1, shuffled2))


def run_tests():
    """Run all tests and print report."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
