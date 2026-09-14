# fsd-scene project article

Static HTML/CSS/JavaScript. The article documents the custom perception attempt, its measured results, and the later Sparse4Dv3 comparison. Described figure and animation blocks are placeholders until source-bound visuals are produced.

The opening prose and the ten-part IPFormer outline follow William's reference report. The IPFormer section expands the outline with explanations checked against the papers and `src/fsd/model.py`. Later article sections are preserved. The embedded opening image is the image supplied in the report; it illustrates the intended scene and is not a project prediction.

From the project root:

```powershell
.venv/Scripts/python.exe -m http.server 8790 --bind 127.0.0.1 --directory website
```

Open http://127.0.0.1:8790/. Edit article text in `index.html`, visual styling in `style.css`, and chapter navigation in `app.js`.

Use `<figure class="media-surface">` with an image and caption for embedded media. Store site images under `assets/` with filenames containing no spaces, and use relative URLs so they work under the GitHub Pages project path. Use `<figure class="code-surface">` with a caption and `<pre><code>` for a code panel; escape HTML characters in code and label pseudocode as such. The code panels scroll horizontally within the article and can be focused with the keyboard. Neither surface changes the existing article grid, sidebar, or typography.

Push website edits to `main` to deploy through `.github/workflows/deploy-pages.yml`. Only `website/` is published. Keep source-report snapshots, imports, and verification outputs in the ignored `artifacts/website-content/` directory.

Desktop uses a sticky chapter index; small screens use an expandable contents menu. Keyboard focus, skip navigation, reduced-motion and print styling are included. Training remains stopped.
