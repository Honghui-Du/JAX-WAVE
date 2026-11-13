"""Verify the public files, check the wave formulation, or install model inputs."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify():
    records = json.loads((ROOT / 'release_manifest.json').read_text())['files_sha256']
    for name, expected in records.items():
        if checksum(ROOT / name) != expected:
            raise ValueError('Checksum mismatch: ' + name)
    print('Verified', len(records), 'repository files.')


def install(archive, output, spec):
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.jaxwave-', dir=output.parent) as temporary:
        temporary = Path(temporary)
        if archive.is_dir():
            joined = temporary / 'data.zip'
            with joined.open('wb') as destination:
                for part in spec['parts']:
                    path = archive / part['filename']
                    if path.stat().st_size != part['bytes'] or checksum(path) != part['sha256']:
                        raise ValueError('Archive part checksum mismatch: ' + path.name)
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            archive = joined
        if archive.stat().st_size != spec['bytes'] or checksum(archive) != spec['sha256']:
            raise ValueError('Archive checksum mismatch.')
        staged = temporary / 'inputs'
        staged.mkdir()
        with ZipFile(archive) as bundle:
            manifests = [name for name in bundle.namelist() if name.endswith('/PACKAGE_MANIFEST.json')]
            if len(manifests) != 1:
                raise ValueError('Expected one dataset manifest in the archive.')
            manifest_name = manifests[0]
            prefix = manifest_name.rsplit('/', 1)[0] + '/'
            records = json.loads(bundle.read(manifest_name))['files_sha256']
            for name, entry in spec['files'].items():
                if Path(name).name != name:
                    raise ValueError('Dataset output names must be plain filenames.')
                matches = [path for path, digest in records.items() if digest == entry['sha256']]
                if not matches:
                    raise ValueError('Requested input is absent from the archive: ' + name)
                with bundle.open(prefix + matches[0]) as source, (staged / name).open('wb') as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                if checksum(staged / name) != entry['sha256']:
                    raise ValueError('Input checksum mismatch: ' + name)
        staged.rename(output)
    print('Data and checkpoints installed in', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'math', 'install'])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
    elif args.command == 'math':
        subprocess.run([sys.executable, str(ROOT / 'scripts/check_math.py')], check=True)
    else:
        if args.archive is None:
            parser.error('install requires --archive PATH')
        install(args.archive, args.output, json.loads((ROOT / 'artifacts.json').read_text()))


if __name__ == '__main__':
    main()
