# fsd-scene project article

Static HTML/CSS/JavaScript. The article documents the custom perception attempt, its measured results, and the later Sparse4Dv3 comparison. Described figure and animation blocks are placeholders until source-bound visuals are produced.

From the project root:

```powershell
.venv/Scripts/python.exe -m http.server 8790 --bind 127.0.0.1 --directory website
```

Open http://127.0.0.1:8790/. Edit article text in `index.html`, visual styling in `style.css`, and chapter navigation in `app.js`.

Desktop uses a sticky chapter index; small screens use an expandable contents menu. Keyboard focus, skip navigation, reduced-motion and print styling are included. Training remains stopped.
