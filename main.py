import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add('--batch_size', defaut=10, type=int)
    parser.add('--epoch',defaut=10,type=int)
    
