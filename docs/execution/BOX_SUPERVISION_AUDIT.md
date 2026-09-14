# Drive0000 full warm-cache box supervision audit

This is a read-only audit of all2102 prepared training timestamps. Original XML,
caches, geometry and training targets are unchanged. No sealed-test content was
inspected. The complete distributions, flagged IDs and every affected frame are
recorded in `artifacts/data-audit/box-quality-0000.json`.

| Class | Box observations | Unique annotation IDs | Frames containing class | Median length,width,height(m) |
|---|---:|---:|---:|---|
| Car | 12065 | 883 | 1732 |4.58,2.04,1.62|
| Pedestrian |492|56|393|0.91,0.89,1.93|

These are annotation-instance IDs; repeated observations are not independent
physicalobjects. The JSON includes both observation-weighted percentiles and
percentiles of per-instance median dimensions.

## Representability and source quality

No cached target exceeds the inference dimensions interval(0.05,30)m or the
initial head's exp(-4)..exp(4) clamp. However, passing numeric bounds does not make
a primitive a precise physicalobject boundary.

Fifty pedestrian observations(10.16%) from7of56 IDs exceed2m horizontalextent:
19002,19005,19016,19021,19024,19025,19027. ID19002 also has2 observations exceeding3m
height. The officialsource—not a cache conversion—contains these examples:

| ID | Native XML axis extents(m) | Affected sampled observations | Observed frames |
|---|---|---:|---|
|19002|1.606,2.327,3.922|2|605,610|
|19005|0.898,2.425,2.122|7|1665-1700|
|19016|2.621,1.665,1.972|11|6390-6450|
|19021|1.144,2.352,1.985|4|7540-7555|
|19024|6.345,0.930,1.901|6|8475-8505|
|19025|4.331,1.112,1.882|16|8655-8790|
|19027|5.547,1.401,2.004|4|9890-9925|

Frame ranges are bounds, not a claim that every intervening sampledframe is
labeled. Exact lists are in the JSON. All7 flagged pedestrian IDs have
`category=instance`, `dynamic=0`, `timestamp=-1` in the official XML. They are
semantic instance primitives with coarse extents. No explicit group category
establishes how many physicalpeople are inside a long primitive. We therefore do
not relabel these as confirmed groups, shrink them, or invent corrected centers.

The image audit shows an approximately4m-tall primitive around a visibleperson
(ID19002), elongated roadside regions for19024/19027, and no isolatedperson
silhouette apparent inside19025 at the chosenframe. Current LiDAR support inside
a primitive does not prove that the primitive tightly bounds one person. A model
that fits these targets can still render physically implausible people.

Cars have132 heuristic shape flags across8 IDs. Of these,69 observations across
5 IDs are normal car shapes with swapped horizontalaxis conventions; the other
63 observations concern oversized/tall boxes13766/13767 and wide dynamicvan13885.
Native height is3.787m for13766 and5.012m for13767. The image audit shows ordinary
vehicles inside substantially oversized vertical bounds. Upright ego-frame
fitting increases some heights further because it encloses tilted primitives.
These are source-quality limitations; no filtering was performed.

## Car-axis semantics and proposed geometry-preserving correction

XML localx is not reliably the physical car's length/forwardaxis. IDs13199,13461,
13491,13624 and13765 have nativeX about1.8-2.0m and nativeY about4.1-5.2m. Images
show normal parkedcars; localx crosses their shortbody axis. Treating that axis as
model length/stockasset forward can produce sideways stretched rendering while
retaining correct cuboid geometry.

For a car with width>length, a possible representation correction is:

```
length_new = width_old
width_new = length_old
yaw_new = wrap_to_pi(yaw_old + pi/2)
```

Center, height and all occupiedgeometry remain unchanged. This was tested without
editing source/targets on all69 affected real observations: maximum symmetric
corner-set error7.1e-15m; minimum BEV/3D IoU above0.999999999999998. The affected
IDs have14,3,15,22,15 observations respectively. None of the99 car observations
in the16-frame overfit block5355-5430 is affected.

This proposal makes the longaxis consistent, but **does not establish vehicle
front versus rear**. The source XML contains transforms and instance metadata,
not a separately verified directionalheading label. Positive longaxis sign can
still differby180degrees; pedestrian yaw does not establish bodyfacing either.
A future decision must distinguish unoriented box-axis accuracy from physical
front-facing accuracy. Sourceannotations and priornamespace should be preserved
if canonicalization is approved. No pipeline change was made during this audit.

## Images and reproduction

`artifacts/data-audit/box-quality-0000-images/` contains4camera GT diagnostic
mosaics for all8 flaggedcar IDs and representativecoarse pedestrians. Orange shows
bounds; cyan shows positiveXML-derived xaxis. These are diagnosticGT artifacts,
never normal replay predictions. Nativefisheye projection is used; straight lines
between projectedcorners are an illustration, not a curvededge projection model.

Reproduce with:

```
.venv/Scripts/python.exe artifacts/data-audit/audit_box_quality.py
.venv/Scripts/python.exe artifacts/data-audit/render_box_quality_audit.py
.venv/Scripts/python.exe artifacts/data-audit/check_car_canonicalization.py
```

The human image interpretation in the finalJSON is an explicit audit addition.

## Drive0002 missing fisheye input coverage

The selectedscene list has2242 timestamps and the two perspectivearchives each
contain2242 selectedmembers. Camera02 has2198 selectedmembers:44 earlyselected
frames4395..4610 step5 are absent. Their perspectiveimages exist. All4 timestamp
textfiles have timestamp entries, so timestamp availability alone is insufficient.

The existing preparation guard checks allfour actualimagefiles andLiDAR before
creatingtargets. These44 frames will enter the manifest exclusions as
`missing_image_or_lidar`, keyed bysequence andframe_id; they cannot silently enter
the fully observed4camera training samples. Downloading every available selected
archive member does not prove every requestedscene input exists.

`artifacts/data-audit/input-coverage-0002.json` records the gap and guard. Camera03
coverage and the finalfull exclusiontotal still require completion of acquisition
and mechanical preparation. Full acquisition was left running unchanged.

## Accepted follow-up

After the read-only audit, the parent accepted the car-only geometry-preserving
canonicalization before baseline training. It is now implemented, with original
manifest/caches preserved under original-axes naming. This document and
box-quality-0000.json retain the pre-correction distributions. See
car-canonicalization-accepted.json for the12557-target source recomputation proof
and architecturev0.2 for the accepted correctness decision. Coarseprimitive flags
remain unfiltered; front/backheading remains unverified.

Later mechanical progress confirms camera03 also selects2198archive members
and its processedprefix is missing the same44earlytimestamps. Thus both fisheyes
share the known gap. The finalexclusion total still awaits complete preparation;
other target-coverage exclusions remain possible.
