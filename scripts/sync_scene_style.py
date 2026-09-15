"""Mirror only shared public renderer modules/assets into the static Pages tree."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
FILES=('road-contours.js','road-surface.js','vehicles.js','assets/traffic-sedan.mesh.js','assets/ego-racer.mesh.js','assets/KENNEY-LICENSE.txt','assets/ASSET-SOURCES.txt')

if __name__=='__main__':
    for relative in FILES:
        source=ROOT/'viewer/scene-style'/relative
        target=ROOT/'website/replay/scene-style'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    print(f'Synchronized {len(FILES)} public renderer files; no model or dataset files copied.')
