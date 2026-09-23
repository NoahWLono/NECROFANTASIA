# NECROFANTASIA ~ Opus 5.5 Remix

An unofficial, non-commercial Touhou Project fan work. The audio and the video are both generated from code; no samples, recordings or official art are used.

- **Music:** *Necrofantasia* (ネクロファンタジア), composed by **ZUN (Team Shanghai Alice)**, from *東方妖々夢 ~ Perfect Cherry Blossom* (2003).
- **Characters:** Touhou Project © ZUN / Team Shanghai Alice. All sprites are drawn procedurally in `src/cameos/`.
- **Note reference:** fan MIDI transcription by Gyana Ren ([VGMusic](https://www.vgmusic.com/file/4ec8919739a19527c18a9ce0bd1c31b7.html)). Only its pitches and rhythms are used. The drums, sound design, structure and mix are new.
- This is not an official production of Anthropic or Team Shanghai Alice.

## Structure (164 BPM, about 5:28)

| Time | Section | What happens |
|---|---|---|
| 0:00 | Boot | Terminal boot, Claude spark bloom, Yukari's gap opens on the title. The harmony plays through a filter sweep. |
| 0:23 | Loop 1 | The full song. Pre-boss dialogue, then Yukari's spell cards. A cameo declares a spell every 4 bars, and Yuyuko and Youmu take the piano section. |
| 3:09 | Interlude | Music-box version of the piano section, with a gallery of "memories". |
| 3:32 | Loop 2 | Key change up a semitone, octave-doubled lead, heavier drums. Two cameos at a time in the final climax. |
| 5:07 | Outro | The whole cast of 50 on screen, credits, and thanks to ZUN. |

## Build

```sh
pip install numpy scipy pillow mido imageio-ffmpeg
python3 src/arrange.py          # -> build/necro.wav + build/timeline.json
python3 src/render.py cache     # -> build/sprites/*.png
python3 src/render.py video     # -> build/necrofantasia_opus55.mp4 (1080p30)
python3 src/render.py still 80  # preview a single frame
```

Bullets fire from the note events in `timeline.json`: melody notes become rings, arpeggios become spirals, crashes become delayed amulet rings, and snares make the active cameo fire an aimed spread.
