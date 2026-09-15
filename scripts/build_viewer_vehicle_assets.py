"""Run with Blender --background --factory-startup --python this.py -- --source ... --output ... .

Refine selected CC0 Quaternius/Kenney GLBs. Export an editable Blender scene
and a compact triangle geometry payload. No ML data or model weights are read.
"""
import argparse
import base64
import json
import math
import sys
from collections import Counter
from array import array
from pathlib import Path

import bpy
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--name', default='traffic-sedan')
    parser.add_argument('--creator', choices=('kenney','quaternius'), default='kenney')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.source.resolve()))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if args.inspect:
        for obj in objects:
            print('ASSET', obj.name, 'dimensions', list(obj.dimensions), 'bounds', [list(obj.matrix_world @ Vector(v)) for v in obj.bound_box], 'materials', [m.name for m in obj.data.materials])
        for image in bpy.data.images:
            print('IMAGE', image.name, list(image.size))
        image=next((image for image in bpy.data.images if image.size[0]==512),None)
        if image is None:return
        pixels=list(image.pixels);iw,ih=image.size
        counts=Counter()
        for obj in objects:
            for poly in obj.data.polygons:
                u,v=obj.data.uv_layers.active.data[poly.loop_indices[0]].uv
                i=4*(int(v*ih)*iw+int(u*iw))
                counts[tuple(round(c,3) for c in pixels[i:i+3])]+=1
        print('PALETTE',counts)
        return
    all_points=[obj.matrix_world @ v.co for obj in objects for v in obj.data.vertices]
    for obj in list(objects):
        if args.creator=='kenney' and obj.name.startswith('wheel'):
            bpy.data.objects.remove(obj,do_unlink=True);objects.remove(obj)
    # Source palette sampled per polygon, replacing the atlas with semantic PBR materials.
    palette = [('body',(.47,.50,.55,1)),('glass',(.10,.14,.19,1)),('tire',(.025,.03,.04,1)),('metal',(.36,.40,.45,1)),('headlight',(.9,.94,1,1)),('taillight',(.42,.025,.035,1))]
    materials = []
    for name, color in palette:
        material=bpy.data.materials.new(name);material.diffuse_color=color
        materials.append(material)
    texture = next((image for image in bpy.data.images if image.size[0]==512),None)
    pixels=list(texture.pixels) if texture else []; iw,ih=texture.size if texture else (0,0)
    lo=Vector([min(p[i] for p in all_points) for i in range(3)])
    hi=Vector([max(p[i] for p in all_points) for i in range(3)])
    center=(lo+hi)/2
    # glTF import is Z-up in Blender; Kenney's longitudinal axis is Y.
    scale=hi-lo
    for obj in objects:
        bpy.context.view_layer.objects.active=obj
        mesh=obj.data;uv=mesh.uv_layers.active
        slots=[]
        for poly in mesh.polygons:
            if args.creator=='quaternius':
                name=mesh.materials[poly.material_index].name.split('.')[0]
                slots.append({'Blue':0,'Windows':1,'Black':2,'Grey':3,'Headlights':4,'TailLights':5}[name])
                continue
            u,v=sum((uv.data[i].uv for i in poly.loop_indices),Vector((0,0)))/len(poly.loop_indices)
            i=4*(min(ih-1,max(0,int(v*ih)))*iw+min(iw-1,max(0,int(u*iw))))
            r,g,b=pixels[i:i+3]
            if max(r,g,b)<.33: slot=2
            elif b>r*1.15 and b>.7: slot=1
            elif min(r,g,b)>.8: slot=1  # white source windows become smoked glass
            elif r>.6 and g>.45 and b<.3: slot=4
            elif r>.35 and r>g*1.8: slot=0
            else: slot=3
            slots.append(slot)
        mesh.materials.clear()
        for material in materials: mesh.materials.append(material)
        for poly,slot in zip(mesh.polygons,slots): poly.material_index=slot
        # Bake world transforms, then normalize so detections retain exact box dimensions.
        matrix=obj.matrix_world.copy()
        for vertex in mesh.vertices:
            p=matrix @ vertex.co-center
            vertex.co=(-p.y/scale.y,p.x/scale.x,p.z/scale.z)
        obj.matrix_world.identity()
        # Weld imported triangle seams before beveling; do not subdivide away silhouette features.
        bpy.ops.object.select_all(action='DESELECT');obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.mesh.remove_doubles(threshold=.00001);bpy.ops.object.mode_set(mode='OBJECT')
        if args.creator!='quaternius' or 'wheel' not in obj.name.lower():
            bevel=obj.modifiers.new('Soft body edges','BEVEL');bevel.width=.008 if args.creator=='quaternius' else .014;bevel.segments=2 if args.creator=='quaternius' else 3;bevel.limit_method='ANGLE';bevel.angle_limit=.45
            bpy.ops.object.modifier_apply(modifier=bevel.name)
        for poly in obj.data.polygons: poly.use_smooth=True
        normal=obj.modifiers.new('Weighted surface normals','WEIGHTED_NORMAL');normal.keep_sharp=True;normal.weight=50
        bpy.ops.object.modifier_apply(modifier=normal.name)
    args.output.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output/(args.name+'.blend')).resolve()))
    # Export material-grouped vertices and Blender's split normals. Round-trip positions
    # remain normalized; original world pose and dimensions are applied only by the viewer.
    chunks=[{'position':array('f'),'normal':array('f')} for _ in materials]
    for obj in objects:
        mesh=obj.data;mesh.calc_loop_triangles()
        for tri in mesh.loop_triangles:
            chunk=chunks[tri.material_index]
            for loop_index in tri.loops:
                loop=mesh.loops[loop_index]
                chunk['position'].extend(mesh.vertices[loop.vertex_index].co)
                chunk['normal'].extend(mesh.corner_normals[loop_index].vector)
    source_label='Quaternius Cars Pack / Car Cz6yDaUcM9' if args.creator=='quaternius' else 'Kenney Car Kit 3.1 / '+args.source.name
    payload={'schema':1,'source':source_label,'license':'CC0-1.0','axis':'x forward, y left, z up','wheels_included':args.creator=='quaternius','parts':[]}
    for (name,_),chunk in zip(palette,chunks):
        if not chunk['position']: continue
        payload['parts'].append({'material':name,**{key:base64.b64encode(value.tobytes()).decode('ascii') for key,value in chunk.items()}})
    (args.output/(args.name+'.mesh.js')).write_text('export default '+json.dumps(payload,separators=(',',':'))+';\n',encoding='utf-8',newline='\n')
    print('EXPORTED',sum(len(c['position'])//9 for c in chunks),'triangles',[(name,len(c['position'])//9) for (name,_),c in zip(palette,chunks)])


if __name__=='__main__':main()
