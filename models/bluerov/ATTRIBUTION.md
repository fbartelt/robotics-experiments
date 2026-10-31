# Attribution and modifications

Original assets and scenario:

- Repository: bvibhav/stonefish_bluerov2
- URL: https://github.com/bvibhav/stonefish_bluerov2
- Retrieved: 27 October 2026
- Repository license: Apache License 2.0 (included as LICENSE_APACHE_2_0.txt)

The assets used are data/bluerov2/bluerov2.obj, br2.png,
bluerov2_wings.obj, cw.obj, ccw.obj, and scenarios/bluerov2.scn.
Original files are included unchanged in original_assets/ and the source scene.

Modifications in this pack: assembled the meshes into a standard GLB, assigned
materials corresponding to the source scene, embedded the original body PNG,
applied the original eight propeller transforms, and rotated the assembled
vehicle 180 degrees about x to use an x-forward/y-left/z-up frame. The Heavy-kit
black material uses explicitly chosen dark-grey PBR values. Added a rendered
preview, geometry metadata, a UAIBot helper, and integration documentation.

BlueROV2 is the name of a Blue Robotics vehicle. No affiliation or endorsement
by the original repository author or Blue Robotics is implied.
