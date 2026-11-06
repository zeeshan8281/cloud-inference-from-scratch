# Bento study video

16 seconds, 1920×1080, 60 fps. Original vector layout and synthesized audio; no copied reference music or branding. The cards use real recorded GPU measurements and explicitly distinguish animation from live throughput.

From this directory, with Node.js, Python 3 and ffmpeg installed:

```bash
npm install
python3 prepare_data.py ../../experiments/numerical-debugger/run-20261106
python3 music.py
npx remotion render index.jsx DivergenceStudy silent.mp4 --codec=h264 --crf=16 --concurrency=4
ffmpeg -i silent.mp4 -i music.wav -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart study.mp4
```

`prepare_data.py` refuses crashed/incomplete runs, changed GPU identity, or a raw/summary checksum mismatch. No synthetic fixture or historical result supplies the published chart values.
