"""Shared constants describing the WildfireSpreadTS tensor layout.

Values come from the official dataset (Zenodo 10.5281/zenodo.8006177) and the
official loader `FireSpreadDataset` (github.com/SebastianGer/WildfireSpreadTS):
23 base channels + land cover expanded to 17 one-hot channels = 40 channels,
with a binary active-fire mask as the last channel after preprocessing.
"""

NUM_BASE_CHANNELS = 23
NUM_LAND_COVER_CLASSES = 17
NUM_CHANNELS = NUM_BASE_CHANNELS + NUM_LAND_COVER_CLASSES  # 40

# Index of the binary active-fire mask channel, added during preprocessing.
ACTIVE_FIRE_CHANNEL = NUM_CHANNELS - 1  # 39
