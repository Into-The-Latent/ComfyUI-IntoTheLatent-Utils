# Shared golden fixture for the Ideogram 4 prompt output — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
# Plain module (no pytest imports) so both the ComfyUI-env node test and the comfy-free core test can use it.
import json

GOLDEN_BOXES = [
    {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "type": "obj", "text": "", "desc": "a crab", "palette": ["#ff0000"]},
    {"x": 0.5, "y": 0.5, "w": 0.25, "h": 0.1, "type": "text", "text": "HELLO", "desc": "title", "palette": []},
]
ELEMS = json.dumps(GOLDEN_BOXES)

# Hand-computed: bbox = round(fraction * 1000), [ymin, xmin, ymax, xmax]; palette upper-cased.
PLAIN_COMPACT = ('{"compositional_deconstruction":{"background":"A beach","elements":['
                 '{"type":"obj","bbox":[200,100,600,400],"desc":"a crab","color_palette":["#FF0000"]},'
                 '{"type":"text","bbox":[500,500,600,750],"text":"HELLO","desc":"title"}]}}')
PHOTO_PREFIX = ('{"high_level_description":"Sunset","style_description":{"aesthetics":"moody","lighting":"",'
                '"photo":"35mm","medium":"film","color_palette":["#112233"]},"compositional_deconstruction"')
ART_PREFIX = ('{"style_description":{"aesthetics":"","lighting":"soft","medium":"","art_style":"ukiyo-e"},'
              '"compositional_deconstruction"')
# coord_mode absolute + bbox_order xy at 1024x768: x*1024 -> 102.4->102, 409.6->410; y*768 -> 153.6->154, 460.8->461
ABS_XY_BOX0 = [102, 154, 410, 461]

# Editor-state variants
UNPLACED = [dict(GOLDEN_BOXES[0], nobbox=True)]
DISABLED_AND_GROUP = [
    dict(GOLDEN_BOXES[0], id="a", disabled=True),
    {"id": "g", "group": True, "name": "G", "x": 0, "y": 0, "w": 1, "h": 1, "type": "obj", "text": "", "desc": "", "palette": []},
    dict(GOLDEN_BOXES[1], id="b", parent="g"),
]
