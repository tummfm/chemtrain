"""Regression tests for JAX force matching on several devices."""

import tempfile

import h5py
import jax
import jax.numpy as jnp
from jax_md import energy, space
import numpy as np
import optax
import pytest
from jax_md_mod import custom_quantity

from chemtrain import parallel
from chemtrain.learn import force_matching
from chemtrain.data.data_loaders import (
    HDF5ParallelDataLoader,
    init_batch_functions,
)
from chemtrain.learn.max_likelihood import shmap_model, shmap_update_fn


@pytest.fixture
def two_device_mesh():
    """Build a data mesh from two visible JAX devices."""
    return jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))


@pytest.mark.parallel
@pytest.mark.jax_multidevice(devices=2)
def test_jax_two_device_update_and_hdf5_loader(two_device_mesh):
    """Check updates and asynchronous HDF5 loading on two JAX devices."""
    context = parallel.resolve_parallelism("jax", mesh=two_device_mesh)

    def model(params, batch):
        return params["weight"] * batch["x"]

    def loss(prediction, batch):
        mse = jnp.mean((prediction - batch["y"]) ** 2)
        return mse, {"mse": mse}

    params = {"weight": jnp.array(0.0)}
    optimizer = optax.sgd(0.1)
    update = shmap_update_fn(model, loss, optimizer, mesh=context.mesh)
    batch = {"x": jnp.arange(8.0), "y": 2.0 * jnp.arange(8.0)}
    out = update(
        params, optimizer.init(params), batch, per_target=True
    )
    assert jnp.isclose(out.loss, 70.0)
    assert jnp.isclose(out.target_losses["mse"], 70.0)
    assert jnp.isclose(out.grad["weight"], -70.0)
    assert jnp.isclose(out.params["weight"], 7.0)
    assert out.loss.sharding.is_fully_replicated
    assert out.target_losses["mse"].sharding.is_fully_replicated
    assert out.grad["weight"].sharding.is_fully_replicated
    assert out.params["weight"].sharding.is_fully_replicated
    assert len(out.params["weight"].addressable_shards) == 2
    np.testing.assert_allclose(
        [shard.data for shard in out.params["weight"].addressable_shards],
        [7.0, 7.0],
    )
    assert out.predictions.sharding.spec == jax.sharding.PartitionSpec("data")
    assert not out.predictions.sharding.is_fully_replicated
    assert [shard.index for shard in out.predictions.addressable_shards] == [
        (slice(0, 4),),
        (slice(4, 8),),
    ]
    np.testing.assert_array_equal(out.predictions, jnp.zeros(8))

    single_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))
    single_update = shmap_update_fn(model, loss, optimizer, mesh=single_mesh)
    single_params, _, single_value, _ = single_update(
        params, optimizer.init(params), batch
    )
    np.testing.assert_allclose(np.asarray(out.loss), np.asarray(single_value))
    np.testing.assert_allclose(
        np.asarray(out.params["weight"]), np.asarray(single_params["weight"])
    )

    with tempfile.NamedTemporaryFile(suffix=".h5") as temporary:
        with h5py.File(temporary.name, "w") as handle:
            handle["x"] = np.arange(32, dtype=np.float32)
        loader = HDF5ParallelDataLoader(temporary.name)
        init, get, release = init_batch_functions(
            loader,
            mb_size=8,
            cache_size=1,
            prefetch=True,
            parallel_context=context,
        )
        state = init(random=False)
        state, loaded_batch = get(state)
        assert loaded_batch["x"].sharding.spec == jax.sharding.PartitionSpec(
            "data"
        )
        np.testing.assert_array_equal(
            loaded_batch["x"], np.arange(8, dtype=np.float32)
        )
        _, loaded_batch = get(state)
        np.testing.assert_array_equal(
            loaded_batch["x"], np.arange(8, 16, dtype=np.float32)
        )
        release()


@pytest.mark.parallel
@pytest.mark.jax_multidevice(devices=2)
def test_cuequivariance_segmented_polynomial_with_vma_check_disabled(
    two_device_mesh,
):
    """Check cuEquivariance sharding and reverse AD without VMA checks."""
    cue = pytest.importorskip("cuequivariance")
    cuex = pytest.importorskip("cuequivariance_jax")
    import chemtrain.compose  # noqa: F401

    polynomial = cue.descriptors.spherical_harmonics(
        cue.SO3(1), [0, 1]
    ).polynomial

    def evaluate(vectors):
        output = jax.ShapeDtypeStruct(
            vectors.shape[:-1] + (polynomial.outputs[0].size,),
            vectors.dtype,
        )
        return cuex.segmented_polynomial(
            polynomial, [vectors], [output], method="naive"
        )[0]

    mapped = jax.jit(jax.shard_map(
        evaluate,
        mesh=two_device_mesh,
        in_specs=jax.sharding.PartitionSpec("data"),
        out_specs=jax.sharding.PartitionSpec("data"),
        check_vma=False,
    ))
    vectors = jnp.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    result = mapped(vectors)

    assert result.sharding.spec == jax.sharding.PartitionSpec("data")
    assert not result.sharding.is_fully_replicated
    assert [shard.data.shape[0] for shard in result.addressable_shards] == [2, 2]

    linear = cue.descriptors.linear(
        cue.Irreps("SO3", "2x0"), cue.Irreps("SO3", "3x0")
    ).polynomial
    weights = jnp.arange(linear.inputs[0].size, dtype=jnp.float32) / 10.0
    features = jnp.arange(
        4 * linear.inputs[1].size, dtype=jnp.float32
    ).reshape(4, linear.inputs[1].size)

    def linear_apply(shared_weights, local_features):
        output = jax.ShapeDtypeStruct(
            (local_features.shape[0], linear.outputs[0].size),
            local_features.dtype,
        )
        return cuex.segmented_polynomial(
            linear,
            [shared_weights, local_features],
            [output],
            method="naive",
        )[0]

    def linear_loss_and_grad(shared_weights, local_features):
        varying_weights = jax.lax.pcast(
            shared_weights, "data", to="varying"
        )
        local_loss, local_grad = jax.value_and_grad(
            lambda value: jnp.mean(
                linear_apply(value, local_features) ** 2
            )
        )(varying_weights)
        return jax.lax.pmean((local_loss, local_grad), "data")

    mapped_loss_and_grad = jax.jit(jax.shard_map(
        linear_loss_and_grad,
        mesh=two_device_mesh,
        in_specs=(
            jax.sharding.PartitionSpec(),
            jax.sharding.PartitionSpec("data"),
        ),
        out_specs=(
            jax.sharding.PartitionSpec(),
            jax.sharding.PartitionSpec(),
        ),
        check_vma=False,
    ))
    distributed_loss, distributed_grad = mapped_loss_and_grad(
        weights, features
    )

    single_loss, single_grad = jax.value_and_grad(
        lambda value: jnp.mean(linear_apply(value, features) ** 2)
    )(weights)
    np.testing.assert_allclose(distributed_loss, single_loss, rtol=1.0e-6)
    np.testing.assert_allclose(distributed_grad, single_grad, rtol=1.0e-6)


@pytest.mark.parallel
@pytest.mark.jax_multidevice(devices=2)
def test_jax_two_device_update_with_neighbor_list(two_device_mesh):
    """Match one- and two-device updates for a neighbor-list model."""
    context = parallel.resolve_parallelism("jax", mesh=two_device_mesh)

    # Build a small physical system whose neighbor list is rebuilt for some
    # snapshots and reused for others.
    box = jnp.asarray(4.0, dtype=jnp.float32)
    displacement, _ = space.periodic(box)
    neighbor_fn, pair_energy = energy.soft_sphere_neighbor_list(
        displacement,
        box,
        sigma=jnp.asarray(0.8, dtype=jnp.float32),
        epsilon=jnp.asarray(1.0, dtype=jnp.float32),
        dr_threshold=0.2,
    )
    positions = jnp.asarray([
        [[0.2, 0.2], [0.9, 0.2], [0.2, 1.0]],
        [[0.2, 0.2], [1.0, 0.2], [0.2, 1.1]],
        [[0.2, 0.2], [0.8, 0.2], [0.2, 0.9]],
        [[0.2, 0.2], [1.1, 0.2], [0.2, 1.0]],
    ])
    initial_neighbor = neighbor_fn.allocate(positions[0])

    def energy_template(params):
        def scaled_energy(position, neighbor, **kwargs):
            del kwargs
            return params["scale"] * pair_energy(position, neighbor)

        return scaled_energy

    reference_energy_fn = energy_template({"scale": jnp.array(1.5)})

    def reference_snapshot(position):
        neighbor = initial_neighbor.update(position)
        return reference_energy_fn(position, neighbor)

    # Generate consistent energy and force targets from the same potential.
    reference_energy, energy_gradient = jax.vmap(
        jax.value_and_grad(reference_snapshot)
    )(positions)
    batch = {
        "R": positions,
        "U": reference_energy,
        "F": -energy_gradient,
    }
    model = force_matching.init_model(
        initial_neighbor,
        {
            "U": custom_quantity.energy_wrapper(None),
            "F": custom_quantity.force_wrapper(None),
        },
        feature_extract_fns={
            "energy_and_force": custom_quantity.energy_force_wrapper(
                energy_template
            )
        },
    )
    loss = force_matching.init_loss_fn(
        gammas={"U": 1.0, "F": 1.0}
    )

    # Compare the complete update, including force derivatives, across meshes.
    params = {"scale": jnp.array(0.75)}
    optimizer = optax.sgd(0.1)
    two_device_update = shmap_update_fn(
        model, loss, optimizer, mesh=context.mesh
    )
    two_params, _, two_loss, two_grad = two_device_update(
        params, optimizer.init(params), batch
    )

    # Inspect the assembled arrays and their physical device-local pieces.
    predictions = shmap_model(model, mesh=context.mesh)(params, batch)
    prediction_shardings = []
    jax.debug.inspect_array_sharding(
        predictions, callback=prediction_shardings.append
    )

    single_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))
    single_device_update = shmap_update_fn(
        model, loss, optimizer, mesh=single_mesh
    )
    single_params, _, single_loss, single_grad = single_device_update(
        params, optimizer.init(params), batch
    )

    assert jnp.isfinite(two_loss)
    np.testing.assert_allclose(np.asarray(two_loss), np.asarray(single_loss))
    np.testing.assert_allclose(
        np.asarray(two_grad["scale"]), np.asarray(single_grad["scale"])
    )
    np.testing.assert_allclose(
        np.asarray(two_params["scale"]), np.asarray(single_params["scale"])
    )
    assert len(prediction_shardings) == 2
    assert all(
        isinstance(sharding, jax.sharding.NamedSharding)
        and sharding.spec[0] == "data"
        and not sharding.is_fully_replicated
        for sharding in prediction_shardings
    )
    assert all(
        len(prediction.addressable_shards) == 2
        and all(shard.data.shape[0] == 2
                for shard in prediction.addressable_shards)
        for prediction in predictions.predictions.values()
    )
