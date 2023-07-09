import argparse

import cv2
import numpy as np

try:
    from imwatermark import WatermarkDecoder
except ImportError as e:
    try:
        # Assume some of the other dependencies such as torch are not fulfilled
        # import file without loading unnecessary libraries.
        import importlib.util
        import sys

        spec = importlib.util.find_spec("imwatermark.maxDct")
        assert spec is not None
        maxDct = importlib.util.module_from_spec(spec)
        sys.modules["maxDct"] = maxDct
        spec.loader.exec_module(maxDct)

        class WatermarkDecoder(object):
            """A minimal version of
            https://github.com/ShieldMnt/invisible-watermark/blob/main/imwatermark/watermark.py
            to only reconstruct bits using dwtDct"""

            def __init__(self, wm_type="bytes", length=0):
                assert wm_type == "bits", "Only bits defined in minimal import"
                self._wmType = wm_type
                self._wmLen = length

            def reconstruct(self, bits):
                if len(bits) != self._wmLen:
                    raise RuntimeError("bits are not matched with watermark length")

                return bits

            def decode(self, cv2Image, method="dwtDct", **configs):
                (r, c, channels) = cv2Image.shape
                if r * c < 256 * 256:
                    raise RuntimeError("image too small, should be larger than 256x256")

                bits = []
                assert method == "dwtDct"
                embed = maxDct.EmbedMaxDct(watermarks=[], wmLen=self._wmLen, **configs)
                bits = embed.decode(cv2Image)
                return self.reconstruct(bits)

    except:
        raise e


# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
MATCH_VALUES = [
    [27, "No watermark detected"],
    [33, "Partial watermark match. Cannot determine with certainty."],
    [
        35,
        (
            "Likely watermarked. In our test 0.02% of real images were "
            'falsely detected as "Likely watermarked"'
        ),
    ],
    [
        49,
        (
            "Very likely watermarked. In our test no real images were "
            'falsely detected as "Very likely watermarked"'
        ),
    ],
]


class GetWatermarkMatch:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(self.watermark)
        self.decoder = WatermarkDecoder("bits", self.num_bits)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Detects the number of matching bits the predefined watermark with one
        or multiple images. Images should be in cv2 format, e.g. h x w x c BGR.

        Args:
            x: ([B], h w, c) in range [0, 255]

        Returns:
           number of matched bits ([B],)
        """
        squeeze = len(x.shape) == 3
        if squeeze:
            x = x[None, ...]

        bs = x.shape[0]
        detected = np.empty((bs, self.num_bits), dtype=bool)
        for k in range(bs):
            detected[k] = self.decoder.decode(x[k], "dwtDct")
        result = np.sum(detected == self.watermark, axis=-1)
        if squeeze:
            return result[0]
        else:
            return result


get_watermark_match = GetWatermarkMatch(WATERMARK_BITS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        nargs="+",
        type=str,
        help="Image files to check for watermarks",
    )
    opts = parser.parse_args()

    print(
        """
        This script tries to detect watermarked images. Please be aware of
        the following:
        - As the watermark is supposed to be invisible, there is the risk that
          watermarked images may not be detected.
        - To maximize the chance of detection make sure that the image has the same
          dimensions as when the watermark was applied (most likely 1024x1024
          or 512x512).
        - Specific image manipulation may drastically decrease the chance that
          watermarks can be detected.
        - There is also the chance that an image has the characteristics of the
          watermark by chance.
        - The watermark script is public, anybody may watermark any images, and
          could therefore claim it to be generated.
        - All numbers below are based on a test using 10,000 images without any
          modifications after applying the watermark.
        """
    )

    for fn in opts.filename:
        image = cv2.imread(fn)
        if image is None:
            print(f"Couldn't read {fn}. Skipping")
            continue

        num_bits = get_watermark_match(image)
        k = 0
        while num_bits > MATCH_VALUES[k][0]:
            k += 1
        print(
            f"{fn}: {MATCH_VALUES[k][1]}",
            f"Bits that matched the watermark {num_bits} from {len(WATERMARK_BITS)}\n",
            sep="\n\t",
        )
