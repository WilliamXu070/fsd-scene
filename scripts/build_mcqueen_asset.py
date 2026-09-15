"""Export the user-supplied McQueen scene for LOCAL preview; no license is implied.

Open the .blend with Blender --background --factory-startup --disable-autoexec,
then --python this file -- --textures ... --output ... . Never run embedded scripts.
"""
import argparse
import base64
import hashlib
import json
import sys
from array import array
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--textures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True, exist_ok=True)
    # Use the rest pose: the website is a static reference car, not a rig player.
    for obj in bpy.context.scene.objects:
        if obj.type=='ARMATURE': obj.data.pose_position='REST'
    bpy.context.view_layer.update()
    deps=bpy.context.evaluated_depsgraph_get()
    meshes=[]
    for obj in list(bpy.context.scene.objects):
        if obj.type!='MESH' or obj.hide_render: continue
        mesh=bpy.data.meshes.new_from_object(obj.evaluated_get(deps), preserve_all_data_layers=True, depsgraph=deps)
        mesh.transform(obj.matrix_world)
        meshes.append((obj.name,mesh))
    points=np.array([list(v.co) for _,mesh in meshes for v in mesh.vertices])
    eye=np.array([list(v.co) for name,mesh in meshes if name.endswith('_0') for v in mesh.vertices]).mean(axis=0)
    # The principal horizontal axis is longitudinal. Eyes disambiguate the front.
    values,vectors=np.linalg.eigh(np.cov(points[:,:2].T))
    front=vectors[:,np.argmax(values)]
    if np.dot(front,eye[:2]-points[:,:2].mean(axis=0))<0: front=-front
    rotation=Matrix(((front[0],front[1],0),(-front[1],front[0],0),(0,0,1)))
    aligned=np.array([list(rotation@Vector(p)) for p in points])
    lo,hi=aligned.min(axis=0),aligned.max(axis=0)
    extent=hi-lo; center=(lo+hi)/2
    dimensions=extent*(4.5/extent[0])
    normalization=Matrix.Diagonal(Vector(1/extent))@rotation
    for name,mesh in meshes:
        mesh.transform(normalization.to_4x4())
        for v in mesh.vertices: v.co-=Vector(center/extent)
        mesh.update();mesh.calc_loop_triangles()
    textures={}
    for key,filename in {'body':'Character_Mcqueen_Body_Decal_OpeningRace.png','eyes':'McQueen_Eyes.jpg','tires':'Character_Mcqueen_Tire_Decal_01_TEX.png'}.items():
        data=(args.textures/filename).read_bytes()
        mime='image/jpeg' if filename.endswith('.jpg') else 'image/png'
        textures[key]='data:'+mime+';base64,'+base64.b64encode(data).decode('ascii')
    material_specs={
        'mcqueen-body':{'map':'body','color':'#ffffff','roughness':.42,'metalness':.05,'clearcoat':.3},
        'mcqueen-dark':{'color':'#171b22','roughness':.65},
        'mcqueen-mouth':{'color':'#171018','roughness':.85},
        'mcqueen-eyes':{'map':'eyes','color':'#ffffff','roughness':.6},
        'mcqueen-tires':{'map':'tires','color':'#ffffff','roughness':.9},
    }
    # The second wheel slot also uses the supplied tyre atlas, not a solid rim colour.
    slot_names={'_1':['mcqueen-body','mcqueen-dark','mcqueen-mouth'], '_0':['mcqueen-eyes'], '_2':['mcqueen-tires','mcqueen-tires']}
    chunks={name:{'position':array('f'),'normal':array('f'),'uv':array('f')} for name in material_specs}
    for name,mesh in meshes:
        slots=slot_names[name[-2:]]
        for tri in mesh.loop_triangles:
            chunk=chunks[slots[tri.material_index]]
            for index in tri.loops:
                loop=mesh.loops[index]
                chunk['position'].extend(mesh.vertices[loop.vertex_index].co)
                chunk['normal'].extend(mesh.corner_normals[index].vector)
                chunk['uv'].extend(mesh.uv_layers.active.data[index].uv)
    payload={'schema':2,'source':'User-supplied lightning-mcqueen.zip / Lightning Mcqueen.blend',
             'license':'Not supplied; local preview only','axis':'x forward, y left, z up','wheels_included':True,
             'dimensions_m':dimensions.tolist(),'character_details_included':True,'materials':material_specs,'textures':textures,
             'parts':[{'material':name,**{k:base64.b64encode(v.tobytes()).decode('ascii') for k,v in chunk.items()}} for name,chunk in chunks.items() if chunk['position']]}
    output=args.output/'ego-mcqueen.mesh.js'
    output.write_text('export default '+json.dumps(payload,separators=(',',':'))+';\n',encoding='utf-8')
    report={'dimensions_m':dimensions.tolist(),'source_forward_xy':front.tolist(),'source_extents':extent.tolist(),
            'triangles':sum(len(c['position'])//9 for c in chunks.values()),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'eye_center_normalized':list(normalization@Vector(eye)-Vector(center/extent))}
    (args.output/'geometry-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('EXPORTED',json.dumps(report))


if __name__=='__main__': main()
