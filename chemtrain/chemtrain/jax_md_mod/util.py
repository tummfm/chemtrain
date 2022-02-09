"""Utility functions helpful for handling variable precision for performance
 increase in GPUs.
 """
from jax import ops
from jax_md.util import f64, f32


def high_precision_segment_sum(data, segment_ids, num_segments=None,
                               out_type=f32, indices_are_sorted=False,
                               unique_indices=False, bucket_size=None):
    """Implements the jax.ops.segment_sum, but casts input to float64 before
    summation and casts back to a target output type afterwards (float32 by
    default). Used to inprove numerical accuracy of summation.
    """
    data = f64(data)
    seg_sum = ops.segment_sum(
        data, segment_ids, num_segments=num_segments,
        indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
        bucket_size=bucket_size
    )
    return out_type(seg_sum)
