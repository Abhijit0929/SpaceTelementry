from src.preprocessing.loader import DataLoader
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.windowing import WindowGenerator


def main():

    loader = DataLoader()

    train, test, labels = loader.load_channel("A-1")

    cleaner = DataCleaner()

    train = cleaner.fit_transform(train)

    test = cleaner.transform(test)

    window = WindowGenerator(window_size=100)

    X_train = window.create(train)

    X_test = window.create(test)

    print("=" * 60)

    print("PIPELINE TEST")

    print("=" * 60)

    print("Train :", X_train.shape)

    print("Test  :", X_test.shape)

    print(labels)


if __name__ == "__main__":
    main()