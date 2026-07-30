from src.preprocessing.loader import DataLoader


def main():

    loader = DataLoader()

    channels = loader.get_channel_names()

    print("=" * 60)
    print("DATA LOADER TEST")
    print("=" * 60)

    print(f"Total Channels : {len(channels)}")
    print(f"First Channel  : {channels[0]}")

    train, test, labels = loader.load_channel(channels[0])

    print("\nTrain Shape :", train.shape)
    print("Test Shape  :", test.shape)

    print("\nGround Truth")

    print(labels)


if __name__ == "__main__":
    main()