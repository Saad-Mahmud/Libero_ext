import json
import math
import shutil
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    EXTENSION_ROOT
    / "libero_custom_object"
    / "assets"
    / "objects"
    / "obvious_hazards"
)
OBJECT_ROOT = EXTENSION_ROOT / "libero_custom_object" / "assets" / "objects"
TURBOSQUID_DYNAMITE_SOURCE_DIR = OBJECT_ROOT / "Dynamite"
TURBOSQUID_PISTOL_SOURCE_DIR = OBJECT_ROOT / "postol_ts"

KENNEY_URL = (
    "https://kenney.nl/media/pages/assets/blaster-kit/"
    "aa06525a20-1753959510/kenney_blaster-kit_2.1.zip"
)
DYNAMITE_URL = "https://www.get3dmodels.com/download/tnt_light_the_fuse_by_get3dmodels.glb"
PIXABAY_BOMB_URL = "https://pixabay.com/3d-models/bomb-fuse-explosive-danger-black-478/"


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def download(url, path, referer=None):
    if path.exists() and path.stat().st_size > 0:
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    if referer is not None:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def primitive_xml(model_name, body_xml, bbox_size, bbox_pos=None):
    if bbox_pos is None:
        bbox_pos = (0.0, 0.0, bbox_size[2])
    top_z = bbox_pos[2] + bbox_size[2]
    radius = max(bbox_size[0], bbox_size[1])
    return f"""<mujoco model="{model_name}">
  <worldbody>
    <body>
      <body name="object">
{body_xml}
        <geom
          name="reg_bbox"
          conaffinity="0"
          contype="0"
          group="3"
          pos="{bbox_pos[0]:.4f} {bbox_pos[1]:.4f} {bbox_pos[2]:.4f}"
          rgba="0 0 0 0"
          size="{bbox_size[0]:.4f} {bbox_size[1]:.4f} {bbox_size[2]:.4f}"
          type="box"
        />
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {top_z:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{radius:.4f} {radius:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
"""


def write_primitive_assets():
    bomb_body = """        <geom conaffinity="0" contype="0" group="1" pos="0 0 0.055" rgba="0.01 0.01 0.012 1" size="0.055" type="sphere" />
        <geom conaffinity="0" contype="0" group="1" fromto="0.025 0 0.105 0.068 0.015 0.145" rgba="0.08 0.055 0.025 1" size="0.006" type="capsule" />
        <geom conaffinity="0" contype="0" group="1" pos="0.074 0.017 0.149" rgba="1.0 0.35 0.02 1" size="0.014" type="sphere" />
        <geom conaffinity="0" contype="0" group="1" pos="0.074 0.017 0.149" rgba="1.0 0.95 0.15 1" size="0.007" type="sphere" />
        <geom density="100" friction="0.95 0.3 0.1" group="0" pos="0 0 0.055" rgba="0.8 0.8 0.8 0.0" size="0.060" type="sphere" />"""
    write_text(
        ASSET_ROOT / "primitive" / "cartoon_bomb" / "model.xml",
        primitive_xml("custom_obvious_cartoon_bomb", bomb_body, (0.080, 0.070, 0.080), (0, 0, 0.080)),
    )

    dynamite_body = """        <geom conaffinity="0" contype="0" group="1" pos="0 -0.030 0.040" quat="0.7071068 0 0.7071068 0" rgba="0.78 0.025 0.018 1" size="0.018 0.070" type="cylinder" />
        <geom conaffinity="0" contype="0" group="1" pos="0 0 0.040" quat="0.7071068 0 0.7071068 0" rgba="0.88 0.030 0.020 1" size="0.018 0.070" type="cylinder" />
        <geom conaffinity="0" contype="0" group="1" pos="0 0.030 0.040" quat="0.7071068 0 0.7071068 0" rgba="0.74 0.020 0.015 1" size="0.018 0.070" type="cylinder" />
        <geom conaffinity="0" contype="0" group="1" pos="-0.035 0 0.040" rgba="0.06 0.06 0.06 1" size="0.007 0.061 0.024" type="box" />
        <geom conaffinity="0" contype="0" group="1" pos="0.035 0 0.040" rgba="0.06 0.06 0.06 1" size="0.007 0.061 0.024" type="box" />
        <geom conaffinity="0" contype="0" group="1" fromto="0.070 0.000 0.060 0.105 0.010 0.095" rgba="0.08 0.055 0.025 1" size="0.004" type="capsule" />
        <geom conaffinity="0" contype="0" group="1" pos="0.110 0.012 0.098" rgba="1.0 0.55 0.05 1" size="0.010" type="sphere" />
        <geom density="100" friction="0.95 0.3 0.1" group="0" pos="0 0 0.040" rgba="0.8 0.8 0.8 0.0" size="0.088 0.060 0.035" type="box" />"""
    write_text(
        ASSET_ROOT / "primitive" / "dynamite_bundle" / "model.xml",
        primitive_xml("custom_obvious_dynamite_bundle", dynamite_body, (0.116, 0.064, 0.060), (0, 0, 0.060)),
    )

    toy_gun_body = """        <geom conaffinity="0" contype="0" group="1" pos="-0.012 0 0.075" rgba="0.08 0.18 0.42 1" size="0.055 0.020 0.024" type="box" />
        <geom conaffinity="0" contype="0" group="1" pos="0.065 0 0.079" quat="0.7071068 0 0.7071068 0" rgba="0.10 0.11 0.13 1" size="0.014 0.052" type="cylinder" />
        <geom conaffinity="0" contype="0" group="1" pos="0.122 0 0.079" quat="0.7071068 0 0.7071068 0" rgba="1.0 0.42 0.02 1" size="0.016 0.009" type="cylinder" />
        <geom conaffinity="0" contype="0" group="1" pos="-0.050 0 0.034" euler="0 0 -0.32" rgba="0.05 0.06 0.08 1" size="0.017 0.019 0.044" type="box" />
        <geom conaffinity="0" contype="0" group="1" pos="-0.030 0 0.105" rgba="0.15 0.35 0.72 1" size="0.024 0.016 0.010" type="box" />
        <geom density="100" friction="0.95 0.3 0.1" group="0" pos="0.016 0 0.065" rgba="0.8 0.8 0.8 0.0" size="0.125 0.030 0.065" type="box" />"""
    write_text(
        ASSET_ROOT / "primitive" / "toy_gun" / "model.xml",
        primitive_xml("custom_obvious_toy_gun", toy_gun_body, (0.132, 0.035, 0.075), (0.016, 0, 0.075)),
    )


def normalize_vertices(vertices, target_longest):
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    spans = [maxs[i] - mins[i] for i in range(3)]
    scale = target_longest / max(spans)
    center_x = (mins[0] + maxs[0]) / 2.0
    center_y = (mins[1] + maxs[1]) / 2.0
    min_z = mins[2]
    return [
        ((x - center_x) * scale, (y - center_y) * scale, (z - min_z) * scale)
        for x, y, z in vertices
    ]


def write_obj(path, vertices, faces):
    lines = ["# Converted/generated visual mesh for LIBERO obvious hazard prop"]
    for x, y, z in vertices:
        lines.append(f"v {x:.7f} {y:.7f} {z:.7f}")
    for face in faces:
        lines.append("f " + " ".join(str(index + 1) for index in face))
    write_text(path, "\n".join(lines) + "\n")


def make_uv_sphere_obj(path, radius=0.060, rings=10, segments=18):
    vertices = [(0.0, 0.0, radius * 2.0)]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        z = radius + radius * math.cos(phi)
        r = radius * math.sin(phi)
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append((r * math.cos(theta), r * math.sin(theta), z))
    vertices.append((0.0, 0.0, 0.0))

    faces = []
    bottom_index = len(vertices) - 1
    for segment in range(segments):
        faces.append((0, 1 + segment, 1 + ((segment + 1) % segments)))
    for ring in range(rings - 2):
        row = 1 + ring * segments
        next_row = row + segments
        for segment in range(segments):
            a = row + segment
            b = row + ((segment + 1) % segments)
            c = next_row + ((segment + 1) % segments)
            d = next_row + segment
            faces.append((a, d, c))
            faces.append((a, c, b))
    last_row = 1 + (rings - 2) * segments
    for segment in range(segments):
        faces.append((bottom_index, last_row + ((segment + 1) % segments), last_row + segment))
    write_obj(path, vertices, faces)


COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def parse_glb(path):
    data = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total_length != len(data):
        raise ValueError(f"Unsupported GLB header: {path}")

    offset = 12
    gltf = None
    bin_chunk = None
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk
    if gltf is None or bin_chunk is None:
        raise ValueError(f"GLB is missing JSON or BIN chunk: {path}")
    return gltf, bin_chunk


def accessor_values(gltf, bin_chunk, accessor_index):
    accessor = gltf["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("Sparse GLB accessors are not supported")
    view = gltf["bufferViews"][accessor["bufferView"]]
    fmt, component_size = COMPONENT_FORMATS[accessor["componentType"]]
    component_count = TYPE_COUNTS[accessor["type"]]
    item_size = component_size * component_count
    stride = view.get("byteStride", item_size)
    base = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    values = []
    for index in range(accessor["count"]):
        item_offset = base + index * stride
        values.append(
            struct.unpack_from("<" + fmt * component_count, bin_chunk, item_offset)
        )
    return values


def identity():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul(a, b):
    return [
        [sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def node_matrix(node):
    if "matrix" in node:
        values = node["matrix"]
        return [
            [values[0], values[4], values[8], values[12]],
            [values[1], values[5], values[9], values[13]],
            [values[2], values[6], values[10], values[14]],
            [values[3], values[7], values[11], values[15]],
        ]

    translation = node.get("translation", [0.0, 0.0, 0.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rotation = [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w, 0],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y, 0],
        [0, 0, 0, 1],
    ]
    scaling = [
        [scale[0], 0, 0, 0],
        [0, scale[1], 0, 0],
        [0, 0, scale[2], 0],
        [0, 0, 0, 1],
    ]
    translated = identity()
    translated[0][3], translated[1][3], translated[2][3] = translation
    return matmul(translated, matmul(rotation, scaling))


def transform_vertex(matrix, vertex):
    x, y, z = vertex
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def glb_to_obj(glb_path, obj_path, target_longest):
    gltf, bin_chunk = parse_glb(glb_path)
    vertices = []
    faces = []

    def add_mesh(mesh_index, transform):
        mesh = gltf["meshes"][mesh_index]
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                continue
            if "POSITION" not in primitive.get("attributes", {}):
                continue
            local_vertices = accessor_values(
                gltf, bin_chunk, primitive["attributes"]["POSITION"]
            )
            base = len(vertices)
            vertices.extend(transform_vertex(transform, vertex[:3]) for vertex in local_vertices)

            if "indices" in primitive:
                indices = [int(value[0]) for value in accessor_values(gltf, bin_chunk, primitive["indices"])]
            else:
                indices = list(range(len(local_vertices)))
            for index in range(0, len(indices), 3):
                if index + 2 < len(indices):
                    faces.append((base + indices[index], base + indices[index + 1], base + indices[index + 2]))

    def walk(node_index, parent_transform):
        node = gltf["nodes"][node_index]
        transform = matmul(parent_transform, node_matrix(node))
        if "mesh" in node:
            add_mesh(node["mesh"], transform)
        for child in node.get("children", []):
            walk(child, transform)

    if "scenes" in gltf:
        scene = gltf["scenes"][gltf.get("scene", 0)]
        for node_index in scene.get("nodes", []):
            walk(node_index, identity())
    else:
        for node_index in range(len(gltf.get("nodes", []))):
            walk(node_index, identity())

    if not vertices or not faces:
        raise ValueError(f"No triangle mesh found in {glb_path}")
    write_obj(obj_path, normalize_vertices(vertices, target_longest), faces)


def obj_to_normalized_obj(source_obj, target_obj, target_longest, axis_order=(0, 1, 2)):
    vertices = []
    faces = []
    for line in source_obj.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            source_vertex = tuple(float(value) for value in parts[1:4])
            vertices.append(tuple(source_vertex[index] for index in axis_order))
        elif parts[0] == "f" and len(parts) >= 4:
            indices = []
            for token in parts[1:]:
                index = int(token.split("/")[0])
                if index < 0:
                    index = len(vertices) + index + 1
                indices.append(index - 1)
            for offset in range(1, len(indices) - 1):
                faces.append((indices[0], indices[offset], indices[offset + 1]))
    if not vertices or not faces:
        raise ValueError(f"No OBJ mesh found in {source_obj}")
    write_obj(target_obj, normalize_vertices(vertices, target_longest), faces)


FBX_SCALAR_FORMATS = {
    "Y": ("h", 2),
    "C": ("?", 1),
    "I": ("i", 4),
    "F": ("f", 4),
    "D": ("d", 8),
    "L": ("q", 8),
}
FBX_ARRAY_FORMATS = {
    "f": ("f", 4),
    "d": ("d", 8),
    "l": ("q", 8),
    "i": ("i", 4),
    "b": ("?", 1),
}


def read_fbx_property(data, offset):
    type_code = chr(data[offset])
    offset += 1

    if type_code in FBX_SCALAR_FORMATS:
        fmt, size = FBX_SCALAR_FORMATS[type_code]
        return struct.unpack_from("<" + fmt, data, offset)[0], offset + size

    if type_code in FBX_ARRAY_FORMATS:
        length, encoding, byte_length = struct.unpack_from("<III", data, offset)
        offset += 12
        raw = data[offset : offset + byte_length]
        offset += byte_length
        if encoding == 1:
            raw = zlib.decompress(raw)
        elif encoding != 0:
            raise ValueError(f"Unsupported FBX array encoding: {encoding}")

        fmt, size = FBX_ARRAY_FORMATS[type_code]
        expected = length * size
        if len(raw) < expected:
            raise ValueError("FBX array is shorter than expected")
        return list(struct.unpack_from("<" + fmt * length, raw, 0)), offset

    if type_code in {"S", "R"}:
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        raw = data[offset : offset + length]
        offset += length
        if type_code == "S":
            return raw.decode("utf-8", "ignore"), offset
        return raw, offset

    raise ValueError(f"Unsupported FBX property type: {type_code}")


def read_fbx_node(data, offset):
    if offset + 13 > len(data):
        return None, offset

    end_offset, property_count, _, name_length = struct.unpack_from("<IIIb", data, offset)
    offset += 13
    if end_offset == 0 and property_count == 0 and name_length == 0:
        return None, offset

    name = data[offset : offset + name_length].decode("utf-8", "ignore")
    offset += name_length
    properties = []
    for _ in range(property_count):
        value, offset = read_fbx_property(data, offset)
        properties.append(value)

    children = []
    while offset < end_offset - 13:
        child, offset = read_fbx_node(data, offset)
        if child is None:
            break
        children.append(child)
    return {"name": name, "properties": properties, "children": children}, end_offset


def iter_fbx_nodes(node, name):
    if node["name"] == name:
        yield node
    for child in node["children"]:
        yield from iter_fbx_nodes(child, name)


def first_child(node, name):
    for child in node["children"]:
        if child["name"] == name:
            return child
    return None


def fbx_to_obj(fbx_path, obj_path, target_longest):
    data = fbx_path.read_bytes()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"Only binary FBX files are supported: {fbx_path}")

    roots = []
    offset = 27
    while offset < len(data) - 13:
        node, offset = read_fbx_node(data, offset)
        if node is None:
            break
        roots.append(node)

    geometries = []
    for root in roots:
        geometries.extend(iter_fbx_nodes(root, "Geometry"))
    if not geometries:
        raise ValueError(f"No Geometry node found in {fbx_path}")

    vertices = []
    faces = []
    for geometry in geometries:
        vertices_node = first_child(geometry, "Vertices")
        indices_node = first_child(geometry, "PolygonVertexIndex")
        if vertices_node is None or indices_node is None:
            continue
        raw_vertices = vertices_node["properties"][0]
        raw_indices = indices_node["properties"][0]
        base = len(vertices)
        vertices.extend(
            tuple(raw_vertices[index : index + 3])
            for index in range(0, len(raw_vertices), 3)
        )

        polygon = []
        for raw_index in raw_indices:
            if raw_index < 0:
                polygon.append(-raw_index - 1)
                for offset in range(1, len(polygon) - 1):
                    faces.append(
                        (
                            base + polygon[0],
                            base + polygon[offset],
                            base + polygon[offset + 1],
                        )
                    )
                polygon = []
            else:
                polygon.append(raw_index)

    if not vertices or not faces:
        raise ValueError(f"No triangle mesh could be extracted from {fbx_path}")
    write_obj(obj_path, normalize_vertices(vertices, target_longest), faces)


def mesh_xml(model_name, mesh_file, rgba, bbox_size, bbox_pos=None, extra_visual_xml=""):
    if bbox_pos is None:
        bbox_pos = (0.0, 0.0, bbox_size[2])
    top_z = bbox_pos[2] + bbox_size[2]
    radius = max(bbox_size[0], bbox_size[1])
    return f"""<mujoco model="{model_name}">
  <asset>
    <mesh name="visual_mesh" file="{mesh_file}" />
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom conaffinity="0" contype="0" group="1" mesh="visual_mesh" rgba="{rgba}" type="mesh" />
{extra_visual_xml}
        <geom
          density="100"
          friction="0.95 0.3 0.1"
          group="0"
          pos="{bbox_pos[0]:.4f} {bbox_pos[1]:.4f} {bbox_pos[2]:.4f}"
          rgba="0.8 0.8 0.8 0.0"
          size="{bbox_size[0]:.4f} {bbox_size[1]:.4f} {bbox_size[2]:.4f}"
          type="box"
        />
        <geom
          name="reg_bbox"
          conaffinity="0"
          contype="0"
          group="3"
          pos="{bbox_pos[0]:.4f} {bbox_pos[1]:.4f} {bbox_pos[2]:.4f}"
          rgba="0 0 0 0"
          size="{bbox_size[0]:.4f} {bbox_size[1]:.4f} {bbox_size[2]:.4f}"
          type="box"
        />
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {top_z:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{radius:.4f} {radius:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
"""


def write_mesh_assets():
    bomb_dir = ASSET_ROOT / "mesh" / "cartoon_bomb"
    make_uv_sphere_obj(bomb_dir / "bomb_body.obj")
    write_text(
        bomb_dir / "model.xml",
        mesh_xml(
            "custom_obvious_mesh_cartoon_bomb",
            "bomb_body.obj",
            "0.01 0.01 0.012 1",
            (0.080, 0.070, 0.080),
            (0, 0, 0.080),
            """        <geom conaffinity="0" contype="0" group="1" fromto="0.025 0 0.105 0.068 0.015 0.145" rgba="0.08 0.055 0.025 1" size="0.006" type="capsule" />
        <geom conaffinity="0" contype="0" group="1" pos="0.074 0.017 0.149" rgba="1.0 0.35 0.02 1" size="0.014" type="sphere" />""",
        ),
    )
    write_text(
        bomb_dir / "SOURCE.txt",
        "Source target: Pixabay cartoon bomb GLB\n"
        f"URL: {PIXABAY_BOMB_URL}\n"
        "License listed by source page: Pixabay Content License\n"
        "Note: the source page blocks automated downloads in this environment, "
        "so this mesh-backed variant uses a generated low-poly visual mesh "
        "matching the requested cartoon-bomb concept.\n",
    )

    dynamite_dir = ASSET_ROOT / "mesh" / "dynamite_bundle"
    source_glb = dynamite_dir / "source" / "tnt_light_the_fuse_by_get3dmodels.glb"
    download(
        DYNAMITE_URL,
        source_glb,
        referer="https://www.get3dmodels.com/tools-and-gadgets/dynamite-stick-bundle/",
    )
    glb_to_obj(source_glb, dynamite_dir / "dynamite_bundle.obj", target_longest=0.190)
    write_text(
        dynamite_dir / "model.xml",
        mesh_xml(
            "custom_obvious_mesh_dynamite_bundle",
            "dynamite_bundle.obj",
            "0.86 0.05 0.03 1",
            (0.105, 0.070, 0.060),
            (0, 0, 0.060),
        ),
    )
    write_text(
        dynamite_dir / "SOURCE.txt",
        "Source: Get3DModels Dynamite Stick Bundle\n"
        "URL: https://www.get3dmodels.com/tools-and-gadgets/dynamite-stick-bundle/\n"
        f"Downloaded GLB: {DYNAMITE_URL}\n"
        "License listed by source page: CC Attribution\n"
        "Author listed by source page: Chenchanchong\n",
    )

    gun_dir = ASSET_ROOT / "mesh" / "toy_gun"
    source_zip = gun_dir / "source" / "kenney_blaster-kit_2.1.zip"
    download(KENNEY_URL, source_zip)
    with zipfile.ZipFile(source_zip) as archive:
        source_obj = "Models/OBJ format/blaster-a.obj"
        temp_obj = gun_dir / "source" / "blaster-a.obj"
        temp_obj.write_bytes(archive.read(source_obj))
        try:
            write_text(gun_dir / "source" / "License.txt", archive.read("License.txt").decode("utf-8", "ignore"))
        except KeyError:
            pass
    obj_to_normalized_obj(temp_obj, gun_dir / "toy_blaster.obj", target_longest=0.210)
    write_text(
        gun_dir / "model.xml",
        mesh_xml(
            "custom_obvious_mesh_toy_gun",
            "toy_blaster.obj",
            "0.08 0.18 0.42 1",
            (0.125, 0.060, 0.080),
            (0, 0, 0.080),
            """        <geom conaffinity="0" contype="0" group="1" pos="0.105 0 0.060" quat="0.7071068 0 0.7071068 0" rgba="1.0 0.42 0.02 1" size="0.014 0.012" type="cylinder" />""",
        ),
    )
    write_text(
        gun_dir / "SOURCE.txt",
        "Source: Kenney Blaster Kit\n"
        "URL: https://kenney.nl/assets/blaster-kit\n"
        f"Downloaded ZIP: {KENNEY_URL}\n"
        "License: CC0 1.0 Universal\n"
        "Attribution: Kenney, attribution not required but appreciated.\n",
    )

    turbosquid_dir = ASSET_ROOT / "mesh" / "turbosquid_dynamite"
    source_fbx = TURBOSQUID_DYNAMITE_SOURCE_DIR / "Dynamite.fbx"
    if source_fbx.exists():
        (turbosquid_dir / "source").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_fbx, turbosquid_dir / "source" / "Dynamite.fbx")
        source_textures = TURBOSQUID_DYNAMITE_SOURCE_DIR / "textures"
        if source_textures.exists():
            target_textures = turbosquid_dir / "source" / "textures"
            target_textures.mkdir(parents=True, exist_ok=True)
            for texture in source_textures.iterdir():
                if texture.is_file():
                    shutil.copy2(texture, target_textures / texture.name)

        fbx_to_obj(
            turbosquid_dir / "source" / "Dynamite.fbx",
            turbosquid_dir / "turbosquid_dynamite.obj",
            target_longest=0.190,
        )
        write_text(
            turbosquid_dir / "model.xml",
            mesh_xml(
                "custom_obvious_mesh_turbosquid_dynamite",
                "turbosquid_dynamite.obj",
                "0.88 0.04 0.025 1",
                (0.105, 0.070, 0.060),
                (0, 0, 0.060),
            ),
        )
        write_text(
            turbosquid_dir / "SOURCE.txt",
            "Source: user-provided TurboSquid dynamite download\n"
            "Original local source: libreo_custom_object/assets/objects/Dynamite/Dynamite.fbx\n"
            "License: TurboSquid/user-provided; verify redistribution rights before publishing this repo.\n"
            "Conversion: binary FBX geometry extracted to normalized OBJ and wrapped for LIBERO.\n",
        )

    if TURBOSQUID_PISTOL_SOURCE_DIR.exists():
        pistol_dir = ASSET_ROOT / "mesh" / "turbosquid_pistol"
        pistol_dir.mkdir(parents=True, exist_ok=True)
        (pistol_dir / "source").mkdir(parents=True, exist_ok=True)
        for filename in ["Pistol_PM.obj", "Pistol_PM.mtl"]:
            source = TURBOSQUID_PISTOL_SOURCE_DIR / filename
            if source.exists():
                shutil.copy2(source, pistol_dir / "source" / filename)
        obj_to_normalized_obj(
            pistol_dir / "source" / "Pistol_PM.obj",
            pistol_dir / "turbosquid_pistol.obj",
            target_longest=0.190,
            axis_order=(2, 1, 0),
        )
        write_text(
            pistol_dir / "model.xml",
            mesh_xml(
                "custom_obvious_mesh_turbosquid_pistol",
                "turbosquid_pistol.obj",
                "0.03 0.035 0.045 1",
                (0.105, 0.075, 0.025),
                (0, 0, 0.025),
            ),
        )
        write_text(
            pistol_dir / "SOURCE.txt",
            "Source: user-provided TurboSquid pistol download\n"
            "Original local source: libero_custom_object/assets/objects/postol_ts/Pistol_PM.obj\n"
            "License: TurboSquid/user-provided; verify redistribution rights before publishing this repo.\n"
            "Conversion: OBJ axis-mapped, normalized, and wrapped for LIBERO as a nonfunctional visual prop.\n",
        )

        bullet_dir = ASSET_ROOT / "mesh" / "turbosquid_bullet"
        bullet_dir.mkdir(parents=True, exist_ok=True)
        (bullet_dir / "source").mkdir(parents=True, exist_ok=True)
        for filename in ["Bullet.obj", "Bullet.mtl"]:
            source = TURBOSQUID_PISTOL_SOURCE_DIR / filename
            if source.exists():
                shutil.copy2(source, bullet_dir / "source" / filename)
        obj_to_normalized_obj(
            bullet_dir / "source" / "Bullet.obj",
            bullet_dir / "turbosquid_bullet.obj",
            target_longest=0.080,
            axis_order=(2, 0, 1),
        )
        write_text(
            bullet_dir / "model.xml",
            mesh_xml(
                "custom_obvious_mesh_turbosquid_bullet",
                "turbosquid_bullet.obj",
                "0.95 0.63 0.20 1",
                (0.045, 0.018, 0.018),
                (0, 0, 0.018),
            ),
        )
        write_text(
            bullet_dir / "SOURCE.txt",
            "Source: user-provided TurboSquid pistol/bullet download\n"
            "Original local source: libero_custom_object/assets/objects/postol_ts/Bullet.obj\n"
            "License: TurboSquid/user-provided; verify redistribution rights before publishing this repo.\n"
            "Conversion: OBJ axis-mapped, enlarged for visibility, normalized, and wrapped for LIBERO as a nonfunctional visual prop.\n",
        )


def main():
    write_primitive_assets()
    write_mesh_assets()
    print(f"wrote obvious hazard assets under {ASSET_ROOT}")


if __name__ == "__main__":
    main()
