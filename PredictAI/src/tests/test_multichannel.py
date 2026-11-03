from src.dataset import NASADataset

dataset = NASADataset()
channels = dataset.get_channel_names()
print(f"Total channels found: {len(channels)}")
print("First 3 channels:", channels[:3])

# Load channel A-1 statistics
train, test, label = dataset.load_channel("A-1")
print("Channel A-1 train shape:", train.shape)
print("Channel A-1 test shape:", test.shape)