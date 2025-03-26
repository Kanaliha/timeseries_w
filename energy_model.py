import argparse

def main():
    print('Entering main')
    parser = argparse.ArgumentParser(description='Energy Model')
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--quantity', required=True, help='Quantity to model (e.g., Consumption)')
    args = parser.parse_args()

    print(args)

if __name__ == '__main__':
    main()