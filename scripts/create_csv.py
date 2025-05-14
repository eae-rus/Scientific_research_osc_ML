import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataflow.raw_to_csv import RawToCSV


def main():
   RawToCSV().create_csv()


if __name__ == '__main__':
    main()