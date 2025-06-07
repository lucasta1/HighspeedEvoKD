import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.nasbench101.build import (
    get_nb101_model,
    get_rnd_nb101_and_acc,
)


def main():
    _, acc, hash = get_rnd_nb101_and_acc()
    print(_)
    print(acc)
    print(hash)

    
if __name__ == "__main__":
    main()
    
    