import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import ClassBiT


def main():
    path = 'test.png'
    bit = ClassBiT.BiT(path, bfeat=True, tfeat=True, unsharpfilter=True, crimminsfilter=True, normalization=True)
    feature = bit.features()
    print(feature)


if __name__ == "__main__" :
    main()
