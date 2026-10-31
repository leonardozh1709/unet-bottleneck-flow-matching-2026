from dataclasses import fields, is_dataclass
from colorama import Fore, Style
import dill

from Utils.paths import OUTPUTS_DIR

def dataclass_to_dict(obj) -> dict:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("Expected a dataclass instance.")

    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def main():
    suffix1 = "base_mini64_constant_lr"
    suffix2 = "base_mini64_constant_lr_reseed1"

    config1_path = OUTPUTS_DIR / suffix1 / f"config_{suffix1}.pkl"
    config2_path = OUTPUTS_DIR / suffix2 / f"config_{suffix2}.pkl"

    with open(config1_path, "rb") as f1:
        dict1 = dataclass_to_dict(dill.load(f1))

    with open(config2_path, "rb") as f2:
        dict2 = dataclass_to_dict(dill.load(f2))

    if list(dict1.keys()) != list(dict2.keys()):
        print("KEYS ARE NOT THE SAME")
        return
    else:
        print("keys match") 

    for k, v1, v2 in zip(dict1.keys(), dict1.values(), dict2.values()):
        if v1 == v2:
            print(str(k) + ": " + Fore.GREEN + str(v1) + " = " + str(v2) + Style.RESET_ALL)
        else:
            print(str(k) + ": " + Fore.RED + str(v1) + " != " + str(v2) + Style.RESET_ALL)

if __name__ == "__main__":
    main()