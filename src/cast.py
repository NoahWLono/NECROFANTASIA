"""Collect cameo characters from the sprite batches and cache their sprites as PNGs."""
import glob
import importlib.util
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, 'build', 'sprites')
SIZE = 512

# appearance order in the video (keys missing from the batches are skipped)
ORDER = [
    'reimu', 'marisa', 'cirno', 'rumia', 'daiyousei', 'meiling', 'patchouli', 'sakuya',
    'remilia', 'flandre', 'chen', 'alice', 'youmu', 'yuyuko', 'ran',
    'wriggle', 'mystia', 'keine', 'tewi', 'reisen', 'eirin', 'kaguya', 'mokou',
    'suika', 'aya', 'yuuka', 'komachi', 'eiki', 'momiji', 'nitori', 'sanae', 'kanako',
    'suwako', 'iku', 'tenshi', 'parsee', 'yuugi', 'satori', 'orin', 'okuu', 'koishi',
    'nazrin', 'kogasa', 'byakuren', 'nue', 'kokoro', 'seija', 'clownpiece', 'hecatia',
]


def _load_batches():
    chars = {}
    for path in sorted(glob.glob(os.path.join(HERE, 'cameos', 'batch_*.py'))):
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:  # a broken batch should not kill the render
            print('!! failed to load', path, e)
            continue
        chars.update(getattr(mod, 'CHARACTERS', {}))
    return chars


def _placeholder(color):
    im = Image.new('RGBA', (SIZE, SIZE))
    d = ImageDraw.Draw(im)
    d.ellipse((SIZE * .25, SIZE * .1, SIZE * .75, SIZE * .6), fill=color + (255,))
    d.polygon([(SIZE * .3, SIZE * .6), (SIZE * .7, SIZE * .6), (SIZE * .8, SIZE * .95),
               (SIZE * .2, SIZE * .95)], fill=color + (255,))
    return im


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    chars = _load_batches()
    meta = {}
    for key, c in chars.items():
        p = os.path.join(CACHE, key + '.png')
        try:
            im = c['draw'](SIZE).convert('RGBA')
            if im.size != (SIZE, SIZE):
                im = im.resize((SIZE, SIZE), Image.LANCZOS)
        except Exception as e:
            print('!! sprite failed', key, e)
            im = _placeholder(tuple(c.get('color', (200, 200, 200))))
        im.save(p)
        meta[key] = {'name': c['name'], 'color': list(c['color'])[:3], 'spell': c['spell']}
    return meta


def load(meta):
    return {k: Image.open(os.path.join(CACHE, k + '.png')).convert('RGBA') for k in meta}


def ordered(meta):
    keys = [k for k in ORDER if k in meta and k != 'yukari']
    keys += sorted(k for k in meta if k not in keys and k != 'yukari')
    return keys
