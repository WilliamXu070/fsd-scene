"""Reproduce the architecture figure as editable SVG and a matching PNG."""
from pathlib import Path
import html, math
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H = 2400, 1320
BG, INK, MUTED, RULE = '#f5f1eb', '#232b30', '#5c666b', '#c9cecb'
BLUE, GREEN, GOLD = '#dce8ef', '#deebe1', '#efe3cc'
im = Image.new('RGB', (W,H), BG)
d = ImageDraw.Draw(im)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<title>Original custom six-camera scene model</title>', '<desc>Image features are lifted into a shared BEV grid, then road and object heads predict scene geometry. Optional late refinement is distinguished from the trained Stage A route.</desc>', f'<rect width="{W}" height="{H}" fill="{BG}"/>']

def font(size, bold=False, serif=False):
    return ImageFont.truetype('C:/Windows/Fonts/' + ('georgia.ttf' if serif else 'arialbd.ttf' if bold else 'arial.ttf'), size)

def text(x,y,s,size=26,color=INK,bold=False,serif=False):
    d.text((x,y),s,font=font(size,bold,serif),fill=color,anchor='lt')
    family = 'Georgia, serif' if serif else 'Arial, sans-serif'
    svg.append(f'<text x="{x}" y="{y}" dominant-baseline="text-before-edge" font-family="{family}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(s)}</text>')

def line(points,color=INK,width=3,dash=False,arrow=False):
    if dash:
        for (x1,y1),(x2,y2) in zip(points,points[1:]):
            length=math.hypot(x2-x1,y2-y1)
            for i in range(0,int(length),16):
                a,b=i/length,min(i+9,length)/length
                d.line((x1+(x2-x1)*a,y1+(y2-y1)*a,x1+(x2-x1)*b,y1+(y2-y1)*b),fill=color,width=width)
    else: d.line(points,fill=color,width=width,joint='curve')
    dashattr=' stroke-dasharray="9 7"' if dash else ''
    svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}"{dashattr}/>')
    if arrow:
        x,y=points[-1]; px,py=points[-2]; a=math.atan2(y-py,x-px)
        pts=[(x,y),(x-13*math.cos(a)+6*math.sin(a),y-13*math.sin(a)-6*math.cos(a)),(x-13*math.cos(a)-6*math.sin(a),y-13*math.sin(a)+6*math.cos(a))]
        d.polygon(pts,fill=color)
        svg.append(f'<polygon points="{" ".join(f"{u},{v}" for u,v in pts)}" fill="{color}"/>')

def rect(x,y,w,h,fill=BG,stroke=RULE,dash=False):
    if dash:
        d.rectangle((x,y,x+w,y+h), fill=fill)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"/>')
        line([(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)],stroke,2,True)
    else:
        d.rounded_rectangle((x,y,x+w,y+h),radius=12,fill=fill,outline=stroke,width=2)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

def box(x,y,w,h,title,rows,fill=BG,dash=False):
    rect(x,y,w,h,fill,dash=dash)
    title_size=28
    while d.textlength(title,font=font(title_size,True)) > w-36: title_size-=1
    text(x+18,y+20,title,title_size,bold=True)
    for i,r in enumerate(rows):
        size=25
        while d.textlength(r,font=font(size)) > w-36: size-=1
        assert size >= 22, (title,r,size)
        text(x+18,y+67+i*35,r,size,color=MUTED)

def arrow(x1,y1,x2,y2,dash=False): line([(x1,y1),(x2,y2)],dash=dash,arrow=True)

text(55,35,'From camera images to a 3D scene',56,serif=True)
text(58,111,'Our original custom pipeline  /  six-camera nuScenes version  /  implementation reference',28,color=MUTED)
line([(55,170),(2345,170)],RULE,2)

text(55,197,'01  IMAGE FEATURES',25,bold=True)
text(865,197,'02  IMAGE → GROUND GEOMETRY',25,bold=True)
text(1545,197,'03  SHARED SCENE FEATURES → PREDICTIONS',25,bold=True)

# Input camera tiles: deliberately schematic, not predictions or dataset imagery.
for i in range(6):
    x=55+(i%2)*98; y=310+(i//2)*68
    rect(x,y,88,56,'#e5e5df')
    text(x+13,y+17,f'CAM {i+1}',18,bold=True)
text(55,250,'Six views',30,bold=True)
text(55,540,'6 × 3 × 256 × 704',25)
text(55,580,'Synchronized RGB',24,color=MUTED)
arrow(251,400,297,400)

box(300,303,240,198,'ResNet-50',['Shared per camera','ImageNet weights','Stride 8 / 16 / 32'],BLUE)
text(305,540,'C3: 512 channels',24,color=MUTED)
text(305,577,'C4: 1,024  ·  C5: 2,048',24,color=MUTED)
arrow(540,400,585,400)

text(590,250,'Feature pyramid',30,bold=True)
rect(590,310,226,64,BLUE); text(607,330,'P3   128 × 32 × 88',24,bold=True)
rect(609,395,207,64,BLUE); text(623,415,'P4   128 × 16 × 44',23)
rect(628,480,188,64,BLUE); text(641,500,'P5   128 × 8 × 22',23)
arrow(794,480,794,461); arrow(794,395,794,376)
text(590,580,'Top-down FPN fusion',24,color=MUTED)

# P3 is the actual input to both heads, not a concatenation of all scales.
line([(816,342),(843,342),(843,322),(870,322)],arrow=True)
line([(843,342),(843,483),(870,483)],arrow=True)
box(870,262,285,128,'Depth probabilities',['64 radial bins: 1–80 m'],BLUE)
box(870,429,285,118,'Image context',['64 channels per pixel'],BLUE)
text(870,580,'Both heads use P3',24,bold=True)
line([(1155,322),(1190,322),(1190,385),(1230,385)],arrow=True)
line([(1155,488),(1190,488),(1190,420),(1230,420)],arrow=True)

box(1230,305,267,208,'Lift + BEV pooling',['Depth-weight context','along camera rays','Pool all six views'],GOLD)
box(1230,550,267,112,'Calibration',['Camera rays → ego'],GOLD)
arrow(1363,550,1363,513)
text(1227,690,'No rich 3D voxel volume',24,color=MUTED)
arrow(1497,400,1540,400)

box(1540,290,280,152,'BEV CNN',['64 → 128 channels','3 residual blocks'],GREEN)
for i in range(2,-1,-1):
    rect(1558+i*9,475-i*9,218,112,GREEN)
for i in range(1,8): line([(1558+i*27,475),(1558+i*27,587)],'#b4c9bc',1)
for i in range(1,4): line([(1558,475+i*28),(1776,475+i*28)],'#b4c9bc',1)
arrow(1680,442,1680,457)
text(1545,613,'128 × 160 × 160',27,bold=True)
text(1545,654,'80 × 80 m; 0.5 m cells',24,color=MUTED)
text(1545,694,'BEV = bird’s-eye view',24,color=MUTED)

# One BEV representation supplies all three heads in parallel.
line([(1795,540),(1843,540),(1843,320),(1880,320)],arrow=True)
line([(1843,540),(1880,540)],arrow=True)
box(1880,250,465,165,'Road head',['Road / sidewalk / other known ground','3 × 160 × 160 logits','Unknown labels ignored in training'],GREEN)
box(1880,451,465,222,'Object heads',['2 center heatmaps: car + pedestrian','Geometry: offsets, z, log dimensions,','sin(yaw), cos(yaw)','Decode up to 200 initial proposals'],GREEN)
text(1899,690,'A  →  Initial boxes + scores',28,bold=True)
text(1899,732,'Stage A uses these without refinement.',24,color=MUTED)

line([(55,782),(2345,782)],RULE,2)
text(55,808,'04  LATE OBJECT REFINEMENT',25,bold=True)
text(550,808,'Implemented extension • Stage B was not reached in the full training run',26,color=MUTED)

# Separate, explicitly optional lower lane; A links to decoded upper-lane proposals.
box(55,934,285,156,'A  Initial proposals',['Boxes + object features','Up to 200 candidates'],GOLD,True)
arrow(340,1005,385,1005,True)
box(385,925,430,180,'Spatial sampling',['Project center + eight box corners','Sample valid P3 / P4 / P5 features','from the current camera views'],BLUE,True)
text(402,879,'Current camera/FPN features ↓',25,color=MUTED)
arrow(815,1005,862,1005,True)
box(862,925,421,180,'Spatial → temporal',['Attention: 128 channels / 4 heads','One late refinement block','Uses aligned historical features'],BLUE,True)
text(870,1140,'↑ Up to 3 prior observations + ego poses',24,color=MUTED)
arrow(1283,1005,1330,1005,True)
box(1330,925,429,180,'Residual prediction',['Correct position, size and yaw','Update object confidence','Keep proposal if all views invalid'],GOLD,True)
arrow(1759,1005,1807,1005,True)
box(1807,925,538,180,'Final object scene',['Filter / suppress duplicates → at most 100','Metric boxes + class scores → tracking','Road predictions remain a separate output'],GREEN,True)

line([(55,1200),(2345,1200)],RULE,2)
text(55,1220,'Solid: Stage A pathway    ·    Dashed: optional Stage B    ·    FPN sizes are per camera; batch dimension omitted',25,color=MUTED)
text(55,1262,'Source: experiments/nuscenes/code/src/fsd/model.py + training configuration  |  verified 13 September 2026',24,color=MUTED)
svg.append('</svg>')
(OUT/'original-pipeline.svg').write_text('\n'.join(svg),encoding='utf-8')
im.save(OUT/'original-pipeline.png')
print(OUT/'original-pipeline.png')
