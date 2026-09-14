# Project documentation website: visual direction v1

User direction: stop training work; document the existing attempt as a project narrative. User writes the content. First deliver an image-generated design reference, not a functioning website.

Reference: https://www.tinytpu.com/ — readable long-form technical explanation, restrained article chrome, illustrated mechanisms.

## Visual system

- Paper #F5F1EB; ink #232323; link blue #315C86; muted green #729283; annotation orange #C8783E; rule gray #D5D0C8.
- Georgia or a similar readable serif for article titles/body; system sans-serif for controls and small captions.
- Left-aligned 720px reading column, centered within a wider page. Figures may expand to 1040px. A quiet desktop chapter index sits outside the reading column; mobile uses a collapsible contents list.
- Spacious paragraphs, visible keyboard focus, simple underlined links, minimal animation; reduced-motion support.

## Page sequence (proposed headings, not authored content)

Title and author -> short introduction -> camera/3D scene figure -> why this project -> first architecture -> training and evaluation -> what failed -> comparison with pretrained Sparse4D -> lessons and next steps -> references/AI-use disclosure.

```text
Project name                                      Code / About

            Title
Contents    Author / date
            Introductory prose

            [source cameras | predicted scene]
            Playback / scrub / caption

            Why this project?
            User-authored paragraphs

            How the first model worked
            [annotated architecture figure]
```

## Interaction and evidence

Source images and predictions share timestamp controls. Label reference/ground-truth diagnostics separately. No ground truth substitutes for predictions. Design mockup visuals are illustrative only; actual site must use exported evidence. Architecture figures can support user-controlled steps, details expand inline, citations link to exact sources. Plot placeholders must be replaced by verified results with model/split/workload context. Keep custom baseline and pretrained Sparse4D results distinct.

Content remains editable Markdown/MDX in a future implementation, separate from reusable figure, callout, citation and replay components. No frontend framework or hosting is selected yet.

## Design review

Cream and serif are intentional because the user supplied this reference. Avoid a SaaS dashboard, decorative statistic cards, gradients, oversized marketing hero, invented benchmark claims and generic navigation. The distinctive element is paired camera-to-scene evidence inside a readable project essay. This is the story of an attempt and its lessons, not a claim of solved autonomous driving.
