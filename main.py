import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from BioInspiredFDesc.src import ClassBit


def main():
    path = './resources/test.png'
    bit = ClassBit.BiT(path, bfeat=True, tfeat=False, unsharpfilter=True, crimminsfilter=False, normalization=False)
    feature = bit.features()
    print(feature)


if __name__ == "__main__":
    main()
