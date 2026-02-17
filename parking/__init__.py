# Compatibility shim: geonetworkx imports shapely.ops.cascaded_union which was
# removed in Shapely 2.0. Patch it back using its replacement (unary_union).
import shapely.ops as _ops

if not hasattr(_ops, "cascaded_union"):
    from shapely import unary_union
    _ops.cascaded_union = unary_union
