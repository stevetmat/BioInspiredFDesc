import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from BioInspiredFDesc import ClassBit


def main():
    path = 'test.png'
    bit = ClassBit.BiT(path, bfeat=True, tfeat=True, unsharpfilter=True, crimminsfilter=False, normalization=False)
    feature = bit.features()
    print(feature)


if __name__ == "__main__":
    main()
