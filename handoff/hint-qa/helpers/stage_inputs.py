"""Stage and hash-check bundled QA inputs into a new HINT checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hint-root', type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parents[1]
    manifest = json.loads((bundle / 'inputs/SHA256.json').read_text())
    for relative, expected in manifest.items():
        source = bundle / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError(f'Bundled input checksum mismatch: {relative}')
    for case in ('beta0p5', 'beta2p5'):
        source = bundle / 'inputs' / case
        target = args.hint_root.resolve() / 'qa-source' / case
        (target / 'inputs').mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            nested = item.name.startswith(('input.', 'wout_', 'ESSOS_'))
            destination = target / 'inputs' / item.name if nested else target / item.name
            if destination.exists() and destination.read_bytes() != item.read_bytes():
                raise FileExistsError(f'Refusing to replace different input: {destination.name}')
            shutil.copy2(item, destination)
    print('Bundled checksums verified; both QA cases staged.')


if __name__ == '__main__':
    main()
