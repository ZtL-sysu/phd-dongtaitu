"""Create the versioned launcher with only reviewed host-recovery references."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    src=ROOT/'edsgn/scripts/launch_p3_coverage_recovery_root_v1.py'
    assert sha(src)=='007427958e9519ced2e4f283f61c1165f686cff0b5abe3143185fca2f23368b4'
    text=src.read_text(encoding='utf-8')
    replacements={
        'protocols/p3-coverage-recovery-queue-v1.json':'protocols/p3-host-recovery-queue-v2.json',
        '534bf70d79a50dbbd533681210a0e1056d802bcc9e32489b4699604e43af8c81':'7752189ff13fad4a69b8792fa4a7010bb468897b37c9c9d26d16563464c8985f',
        'reviews/p3-coverage-recovery-root-acceptance-v1.json':'reviews/p3-host-recovery-root-acceptance-v2.json',
        'P3 baseline coverage recovery admission':'P3 baseline host recovery admission',
        'logs/p3-coverage-recovery-prelaunch-v1.json':'logs/p3-host-recovery-prelaunch-v2.json',
        'logs/p3-coverage-recovery-controller-result-v1.json':'logs/p3-host-recovery-controller-result-v2.json',
    }
    for before,after in replacements.items():
        assert text.count(before)==1,(before,text.count(before))
        text=text.replace(before,after)
    dest=ROOT/'edsgn/scripts/launch_p3_host_recovery_root_v2.py'
    with dest.open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({'launcher_sha256':sha(dest),'path':str(dest),'replacement_count':len(replacements)}))
if __name__=='__main__':main()
