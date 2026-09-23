# Cameo sprite spec

Each batch file `src/cameos/batch_X.py` exports:

```python
CHARACTERS = {
  "reimu": {
     "name": "Reimu Hakurei",          # display name
     "color": (230, 40, 50),           # signature RGB, used for bullets/name card
     "spell": 'Dream Sign "Fantasy Seal of 1M Tokens"',  # parody spell card, Claude/AI themed
     "draw": draw_reimu,               # fn(size:int=384) -> PIL.Image RGBA, transparent bg
  }, ...
}
```

Rules
- Only numpy + Pillow. No files, no network, no fonts. Deterministic.
- Draw at 4x size then downsample with LANCZOS for antialiasing.
- Style: cute chibi, full body, big head (~45% of height), facing viewer, centered, fits inside the square with ~6% margin. Clean flat shading + one shade tone + thin dark outline (draw a slightly bigger dark shape under each part). Eyes with highlight.
- Each character MUST be recognizable by signature traits: hair color/style, outfit colors, iconic accessory (Reimu's big red bow + gohei, Marisa's witch hat + broom, Sakuya's maid headband + knives, Remilia bat wings + mob cap, Flandre crystal wings, Cirno ice wings + blue bow, etc.).
- Spell names: witty parody mixing the character's real spell-card style with Claude / LLM concepts (tokens, context window, attention, gradient descent, hallucination, tool use, RLHF, constitutions, prompts). Keep them short (< 48 chars).
- Put a `if __name__ == "__main__":` block that renders every character into a contact sheet at `build/sheet_<batch>.png` so you can visually check it (view the PNG with the Read tool and iterate until each character is clearly recognizable).
