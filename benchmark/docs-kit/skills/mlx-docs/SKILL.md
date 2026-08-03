---
name: mlx-docs
description: |
  Upstream MLX API documentation for arrays, custom Metal kernels, fast fused
  operations, quantisation, compilation, function transforms, and memory and
  device management. Use when writing or debugging MLX code on Apple silicon,
  or when the user mentions "MLX", "mx.fast", "metal_kernel", "quantize",
  "mx.compile", "value_and_grad", or "mx.metal".
---

# MLX API documentation

Docstrings from the installed MLX, concatenated in no particular order.

## mx.fast.metal_kernel

```
metal_kernel(name: str, input_names: collections.abc.Sequence[str], output_names: collections.abc.Sequence[str], source: str, header: str = '', ensure_row_contiguous: bool = True, atomic_outputs: bool = False, compile_options: object | None = None) -> object

A jit-compiled custom Metal kernel defined from a source string.

Full documentation: :ref:`custom_metal_kernels`.

Args:
  name (str): Name for the kernel.
  input_names (List[str]): The parameter names of the inputs in the
     function signature.
  output_names (List[str]): The parameter names of the outputs in the
     function signature.
  source (str): Source code. This is the body of a function in Metal,
     the function signature will be automatically generated.
  header (str): Header source code to include before the main function.
     Useful for helper functions or includes that should live outside of
     the main function body.
  ensure_row_contiguous (bool): Whether to ensure the inputs are row contiguous
     before the kernel runs. Default: ``True``.
  atomic_outputs (bool): Whether to use atomic outputs in the function signature
     e.g. ``device atomic<float>``. Default: ``False``.
  compile_options (dict, optional): Options to compile the Metal kernel
     with. Supported options:

     * ``"math_mode"``: The Metal math mode: ``"safe"``, ``"relaxed"``,
       or ``"fast"``. ``"safe"`` preserves IEEE behavior for special
       values such as ``exp(-inf) == 0``. Default: ``"safe"``.

Returns:
  Callable ``metal_kernel``.

Example:

  .. code-block:: python

    def exp_elementwise(a: mx.array):
        source = '''
            uint elem = thread_position_in_grid.x;
            T tmp = inp[elem];
            out[elem] = metal::exp(tmp);
        '''

        kernel = mx.fast.metal_kernel(
            name="myexp",
            input_names=["inp"],
            output_names=["out"],
            source=source
        )
        outputs = kernel(
            inputs=[a],
            template=[("T", mx.float32)],
            grid=(a.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[a.shape],
            output_dtypes=[a.dtype],
            verbose=True,
        )
        return outputs[0]

    a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
    b = exp_elementwise(a)
    assert mx.allclose(b, mx.exp(a))
```

## mx.fast.scaled_dot_product_attention

```
scaled_dot_product_attention(q: array, k: array, v: array, *, scale: float,  mask: Union[None, str, array] = None, sinks: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array

A fast implementation of multi-head attention: ``O = softmax(Q @ K.T, dim=-1) @ V``.

Supports:

* `Multi-Head Attention <https://arxiv.org/abs/1706.03762>`_
* `Grouped Query Attention <https://arxiv.org/abs/2305.13245>`_
* `Multi-Query Attention <https://arxiv.org/abs/1911.02150>`_

.. note::

  * The softmax operation is performed in ``float32`` regardless of
    the input precision.
  * For Grouped Query Attention and Multi-Query Attention, the ``k``
    and ``v`` inputs should not be pre-tiled to match ``q``.

In the following the dimensions are given by:

* ``B``: The batch size.
* ``N_q``: The number of query heads.
* ``N_kv``: The number of key and value heads.
* ``T_q``: The number of queries per example.
* ``T_kv``: The number of keys and values per example.
* ``D``: The per-head dimension.

Args:
    q (array): Queries with shape ``[B, N_q, T_q, D]``.
    k (array): Keys with shape ``[B, N_kv, T_kv, D]``.
    v (array): Values with shape ``[B, N_kv, T_kv, D]``.
    scale (float): Scale for queries (typically ``1.0 / sqrt(q.shape(-1)``).
    mask (str or array, optional): The mask to apply to the
       query-key scores. The mask can be an array or a string indicating
       the mask type. The only supported string type is ``"causal"``. If
       the mask is an array it can be a boolean or additive mask. The mask
       can have at most 4 dimensions and must be broadcast-compatible with
       the shape ``[B, N, T_q, T_kv]``. If an additive mask is given its
       type must promote to the promoted type of ``q``, ``k``, and ``v``.
       The ``"causal"`` mask uses lower-right alignment where the
       last query aligns with the last key.
    sinks (array, optional): An optional array of attention sinks.
       Default: ``None``.

Returns:
    array: The output array.

Example:

  .. code-block:: python

    B = 2
    N_q = N_kv = 32
    T_q = T_kv = 1000
    D = 128

    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_kv, T_kv, D))
    v = mx.random.normal(shape=(B, N_kv, T_kv, D))
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
```

## mx.fast.rope

```
rope(a: array, dims: int, *, traditional: bool, base: Optional[float], scale: float, offset: Union[int, array], freqs: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array

Apply rotary positional encoding to the input.

The input is expected to be at least 3D with shape ``(B, *, T, D)`` where:
  * ``B`` is the batch size.
  * ``T`` is the sequence length.
  * ``D`` is the feature dimension.

Args:
    a (array): The input array.
    dims (int): The feature dimensions to be rotated. If the input feature
      is larger than dims then the rest is left unchanged.
    traditional (bool): If set to ``True`` choose the traditional
      implementation which rotates consecutive dimensions.
    base (float, optional): The base used to compute angular frequency for
      each dimension in the positional encodings. Exactly one of ``base`` and
      ``freqs`` must be ``None``.
    scale (float): The scale used to scale the positions.
    offset (int or array): The position offset to start at. If an
      :obj:`array` is given it can be a scalar or vector of ``B``
      offsets for each example in the batch.
    freqs (array, optional): Optional frequencies to use with RoPE.
      If set, the ``base`` parameter must be ``None``. Default: ``None``.

Returns:
    array: The output array.
```

## mx.fast.rms_norm

```
rms_norm(x: array, weight: Optional[array], eps: float, *, stream: Union[None, Stream, Device] = None) -> array

Root Mean Square normalization (RMS norm).

The normalization is with respect to the last axis of the input ``x``.

Args:
    x (array): Input array.
    weight (array, optional): A multiplicative weight to scale the result by.
      The ``weight`` should be one-dimensional with the same size
      as the last axis of ``x``. If set to ``None`` then no scaling happens.
    eps (float): A small additive constant for numerical stability.

Returns:
    array: The output array.
```

## mx.fast.layer_norm

```
layer_norm(x: array, weight: Optional[array], bias: Optional[array], eps: float, *, stream: Union[None, Stream, Device] = None) -> array

Layer normalization.

The normalization is with respect to the last axis of the input ``x``.

Args:
    x (array): Input array.
    weight (array, optional): A multiplicative weight to scale the result by.
      The ``weight`` should be one-dimensional with the same size
      as the last axis of ``x``. If set to ``None`` then no scaling happens.
    bias (array, optional): An additive offset to be added to the result.
      The ``bias`` should be one-dimensional with the same size
      as the last axis of ``x``. If set to ``None`` then no translation happens.
    eps (float): A small additive constant for numerical stability.

Returns:
    array: The output array.
```

## mx.quantize

```
quantize(w: array, /, group_size: Optional[int] = None, bits: Optional[int] = None, mode: str = 'affine', *, global_scale: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> tuple[array, array, array]

Quantize the array ``w``.

Note, every ``group_size`` elements in a row of ``w`` are quantized
together. Hence, the last dimension of ``w`` should be divisible by
``group_size``.

.. warning::

  ``quantize`` only supports inputs with two or more dimensions with
  the last dimension divisible by ``group_size``

The supported quantization modes are ``"affine"``, ``"mxfp4"``,
``"mxfp8"``, and ``"nvfp4"``. They are described in more detail below.

Args:
  w (array): Array to be quantized
  group_size (int, optional): The size of the group in ``w`` that shares a
    scale and bias. See supported values and defaults in the
    :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
  bits (int, optional): The number of bits occupied by each element of
    ``w`` in the quantized array. See supported values and defaults in the
    :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
  mode (str, optional): The quantization mode. Default: ``"affine"``.
  global_scale (array, optional): The per-input float32 scale used for
    ``"nvfp4"`` quantization if provided. Default: ``None``.

Returns:
  tuple: A tuple with either two or three elements containing:

  * w_q (array): The quantized version of ``w``
  * scales (array): The quantization scales
  * biases (array): The quantization biases (returned for ``mode=="affine"``).

Notes:
  .. _quantize-modes:

  .. table:: Quantization modes

    ======  ======================   ==========================  =============  =====
    mode    group size               bits                        scale type     bias
    ======  ======================   ==========================  =============  =====
    affine  32, 64\ :sup:`*`, 128    2, 3, 4\ :sup:`*`, 5, 6, 8  same as input  yes
    mxfp4   32\ :sup:`*`             4\ :sup:`*`                 e8m0           no
    mxfp8   32\ :sup:`*`             8\ :sup:`*`                 e8m0           no
    nvfp4   16\ :sup:`*`             4\ :sup:`*`                 e4m3           no
    ======  ======================   ==========================  =============  =====

  :sup:`*` indicates the default value when unspecified.

  The ``"affine"`` mode quantizes groups of :math:`g` consecutive
  elements in a row of ``w``. For each group the quantized
  representation of each element :math:`\hat{w_i}` is computed as follows:

  .. math::

    \begin{aligned}
      \alpha &= \max_i w_i \\
      \beta &= \min_i w_i \\
      s &= \frac{\alpha - \beta}{2^b - 1} \\
      \hat{w_i} &= \textrm{round}\left( \frac{w_i - \beta}{s}\right).
    \end{aligned}

  After the above computation, :math:`\hat{w_i}` fits in :math:`b` bits
  and is packed in an unsigned 32-bit integer from the lower to upper
  bits. For instance, for 4-bit quantization we fit 8 elements in an
  unsigned 32 bit integer where the 1st element occupies the 4 least
  significant bits, the 2nd bits 4-7 etc.

  To dequantize the elements of ``w``, we also save :math:`s` and
  :math:`\beta` which are the returned ``scales`` and
  ``biases`` respectively.

  The ``"mxfp4"``, ``"mxfp8"``, and ``"nvfp4"`` modes similarly
  quantize groups of :math:`g` elements of ``w``. For the ``"mx"``
  modes, the group size must be ``32``.  For ``"nvfp4"`` the group
  size must be 16. The elements are quantized to 4-bit or 8-bit
  precision floating-point values: E2M1 for ``"fp4"`` and E4M3 for
  ``"fp8"``. There is a shared 8-bit scale per group. The ``"mx"``
  modes use an E8M0 scale and the ``"nv"`` mode uses an E4M3 scale.
  Unlike ``affine`` quantization, these modes does not have a bias
  value.

  More details on the ``"mx"`` formats can
  be found in the `specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_.
```

## mx.dequantize

```
dequantize(w: array, /, scales: array, biases: Optional[array] = None, group_size: Optional[int] = None, bits: Optional[int] = None, mode: str = 'affine', global_scale: Optional[array] = None, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array

Dequantize the matrix ``w`` using quantization parameters.

Args:
  w (array): Matrix to be dequantized
  scales (array): The scales to use per ``group_size`` elements of ``w``.
  biases (array, optional): The biases to use per ``group_size``
     elements of ``w``. Default: ``None``.
  group_size (int, optional): The size of the group in ``w`` that shares a
    scale and bias. See supported values and defaults in the
    :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
  bits (int, optional): The number of bits occupied by each element of
    ``w`` in the quantized array. See supported values and defaults in the
    :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
  global_scale (array, optional): The per-input float32 scale used for
    ``"nvfp4"`` quantization if provided. Default: ``None``.
  dtype (Dtype, optional): The data type of the dequantized output. If
    ``None`` the return type is inferred from the scales and biases
    when possible and otherwise defaults to ``bfloat16``.
    Default: ``None``.
  mode (str, optional): The quantization mode. Default: ``"affine"``.

Returns:
  array: The dequantized version of ``w``

Notes:
  The currently supported quantization modes are ``"affine"``,
  ``"mxfp4``, ``"mxfp8"``, and ``"nvfp4"``.

  For ``affine`` quantization, given the notation in :func:`quantize`,
  we compute :math:`w_i` from :math:`\hat{w_i}` and corresponding :math:`s`
  and :math:`\beta` as follows

  .. math::

    w_i = s \hat{w_i} + \beta
```

## mx.quantized_matmul

```
quantized_matmul(x: array, w: array, /, scales: array, biases: Optional[array] = None, transpose: bool = True, group_size: Optional[int] = None, bits: Optional[int] = None, mode: str = 'affine', *, stream: Union[None, Stream, Device] = None) -> array

Perform the matrix multiplication with the quantized matrix ``w``. The
quantization uses one floating point scale and bias per ``group_size`` of
elements. Each element in ``w`` takes ``bits`` bits and is packed in an
unsigned 32 bit integer.

Args:
  x (array): Input array
  w (array): Quantized matrix packed in unsigned integers
  scales (array): The scales to use per ``group_size`` elements of ``w``
  biases (array, optional): The biases to use per ``group_size``
    elements of ``w``. Default: ``None``.
  transpose (bool, optional): Defines whether to multiply with the
    transposed ``w`` or not, namely whether we are performing
    ``x @ w.T`` or ``x @ w``. Default: ``True``.
  group_size (int, optional): The size of the group in ``w`` that shares a
    scale and bias. See supported values and defaults in the
    :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
  bits (int, optional): The number of bits occupied by each element of
    ``w`` in the quantized array. See supported values and defaults in the
    :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
  mode (str, optional): The quantization mode. Default: ``"affine"``.

Returns:
  array: The result of the multiplication of ``x`` with ``w``.
```

## mx.compile

```
compile(fun: Callable[P, R], inputs: Optional[object] = None, outputs: Optional[object] = None, shapeless: bool = False) -> Callable[P, R]

Returns a compiled function which produces the same output as ``fun``.

Args:
    fun (Callable): A function which takes a variable number of
      :class:`array` or trees of :class:`array` and returns
      a variable number of :class:`array` or trees of :class:`array`.
    inputs (list or dict, optional): These inputs will be captured during
      the function compilation along with the inputs to ``fun``. The ``inputs``
      can be a :obj:`list` or a :obj:`dict` containing arbitrarily nested
      lists, dictionaries, or arrays. Leaf nodes that are not
      :obj:`array` are ignored. Default: ``None``
    outputs (list or dict, optional): These outputs will be captured and
      updated in a compiled function. The ``outputs`` can be a
      :obj:`list` or a :obj:`dict` containing arbitrarily nested lists,
      dictionaries, or arrays. Leaf nodes that are not :obj:`array` are ignored.
      Default: ``None``
    shapeless (bool, optional): A function compiled with the ``shapeless``
      option enabled will not be recompiled when the input shape changes. Not all
      functions can be compiled with ``shapeless`` enabled. Attempting to compile
      such functions with shapeless enabled will throw. Note, changing the number
      of dimensions or type of any input will result in a recompilation even with
      ``shapeless`` set to ``True``. Default: ``False``

Returns:
    Callable: A compiled function which has the same input arguments
    as ``fun`` and returns the same output(s).
```

## mx.eval

```
eval(*args) -> None

Evaluate an :class:`array` or tree of :class:`array`.

Args:
    *args (arrays or trees of arrays): Each argument can be a single array
      or a tree of arrays. If a tree is given the nodes can be a Python
      :class:`list`, :class:`tuple` or :class:`dict`. Leaves which are not
      arrays are ignored.
```

## mx.value_and_grad

```
value_and_grad(fun: Callable[P, R], argnums: Optional[Union[int, Sequence[int]]] = None, argnames: Union[str, Sequence[str]] = []) -> Callable[P, Tuple[R, Any]]

Returns a function which computes the value and gradient of ``fun``.

The function passed to :func:`value_and_grad` should return either
a scalar loss or a tuple in which the first element is a scalar
loss and the remaining elements can be anything.

.. code-block:: python

    import mlx.core as mx

    def mse(params, inputs, targets):
        outputs = forward(params, inputs)
        lvalue = (outputs - targets).square().mean()
        return lvalue

    # Returns lvalue, dlvalue/dparams
    lvalue, grads = mx.value_and_grad(mse)(params, inputs, targets)

    def lasso(params, inputs, targets, a=1.0, b=1.0):
        outputs = forward(params, inputs)
        mse = (outputs - targets).square().mean()
        l1 = mx.abs(outputs - targets).mean()

        loss = a*mse + b*l1

        return loss, mse, l1

    (loss, mse, l1), grads = mx.value_and_grad(lasso)(params, inputs, targets)

Args:
    fun (Callable): A function which takes a variable number of
      :class:`array` or trees of :class:`array` and returns
      a scalar output :class:`array` or a tuple the first element
      of which should be a scalar :class:`array`.
    argnums (int or list(int), optional): Specify the index (or indices)
      of the positional arguments of ``fun`` to compute the gradient
      with respect to. If neither ``argnums`` nor ``argnames`` are
      provided ``argnums`` defaults to ``0`` indicating ``fun``'s first
      argument.
    argnames (str or list(str), optional): Specify keyword arguments of
      ``fun`` to compute gradients with respect to. It defaults to [] so
      no gradients for keyword arguments by default.

Returns:
    Callable: A function which returns a tuple where the first element
    is the output of `fun` and the second element is the gradients w.r.t.
    the loss.
```

## mx.grad

```
grad(fun: Callable[P, R], argnums: Optional[Union[int, Sequence[int]]] = None, argnames: Union[str, Sequence[str]] = []) -> Callable[P, Any]

Returns a function which computes the gradient of ``fun``.

Args:
    fun (Callable): A function which takes a variable number of
      :class:`array` or trees of :class:`array` and returns
      a scalar output :class:`array`.
    argnums (int or list(int), optional): Specify the index (or indices)
      of the positional arguments of ``fun`` to compute the gradient
      with respect to. If neither ``argnums`` nor ``argnames`` are
      provided ``argnums`` defaults to ``0`` indicating ``fun``'s first
      argument.
    argnames (str or list(str), optional): Specify keyword arguments of
      ``fun`` to compute gradients with respect to. It defaults to [] so
      no gradients for keyword arguments by default.

Returns:
    Callable: A function which has the same input arguments as ``fun`` and
    returns the gradient(s).
```

## mx.vmap

```
vmap(fun: Callable[P, R], in_axes: object = 0, out_axes: object = 0) -> Callable[P, R]

Returns a vectorized version of ``fun``.

Args:
    fun (Callable): A function which takes a variable number of
      :class:`array` or a tree of :class:`array` and returns
      a variable number of :class:`array` or a tree of :class:`array`.
    in_axes (int, optional): An integer or a valid prefix tree of the
      inputs to ``fun`` where each node specifies the vmapped axis. If
      the value is ``None`` then the corresponding input(s) are not vmapped.
      Defaults to ``0``.
    out_axes (int, optional): An integer or a valid prefix tree of the
      outputs of ``fun`` where each node specifies the vmapped axis. If
      the value is ``None`` then the corresponding outputs(s) are not vmapped.
      Defaults to ``0``.

Returns:
    Callable: The vectorized function.
```

## mx.take

```
take(a: array, /, indices: Union[int, array], axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array

Take elements along an axis.

The elements are taken from ``indices`` along the specified axis.
If the axis is not specified the array is treated as a flattened
1-D array prior to performing the take.

As an example, if the ``axis=1`` this is equivalent to ``a[:, indices, ...]``.

Args:
    a (array): Input array.
    indices (int or array): Integer index or input array with integral type.
    axis (int, optional): Axis along which to perform the take. If unspecified
      the array is treated as a flattened 1-D vector.

Returns:
    array: The indexed values of ``a``.
```

## mx.where

```
where(condition: Union[scalar, array], x: Union[scalar, array], y: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array

Select from ``x`` or ``y`` according to ``condition``.

The condition and input arrays must be the same shape or
broadcastable with each another.

Args:
  condition (array): The condition array.
  x (array): The input selected from where condition is ``True``.
  y (array): The input selected from where condition is ``False``.

Returns:
    array: The output containing elements selected from
    ``x`` and ``y``.
```

## mx.matmul

```
matmul(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array

Matrix multiplication.

Perform the (possibly batched) matrix multiplication of two arrays. This function supports
broadcasting for arrays with more than two dimensions.

- If the first array is 1-D then a 1 is prepended to its shape to make it
  a matrix. Similarly if the second array is 1-D then a 1 is appended to its
  shape to make it a matrix. In either case the singleton dimension is removed
  from the result.
- A batched matrix multiplication is performed if the arrays have more than
  2 dimensions.  The matrix dimensions for the matrix product are the last
  two dimensions of each input.
- All but the last two dimensions of each input are broadcast with one another using
  standard numpy-style broadcasting semantics.

Args:
    a (array): Input array or scalar.
    b (array): Input array or scalar.

Returns:
    array: The matrix product of ``a`` and ``b``.
```

## mx.addmm

```
addmm(c: array, a: array, b: array, /, alpha: float = 1.0, beta: float = 1.0,  *, stream: Union[None, Stream, Device] = None) -> array

Matrix multiplication with addition and optional scaling.

Perform the (possibly batched) matrix multiplication of two arrays and add to the result
with optional scaling factors.

Args:
    c (array): Input array or scalar.
    a (array): Input array or scalar.
    b (array): Input array or scalar.
    alpha (float, optional): Scaling factor for the
        matrix product of ``a`` and ``b`` (default: ``1``)
    beta (float, optional): Scaling factor for ``c`` (default: ``1``)

Returns:
    array: ``alpha * (a @ b)  + beta * c``
```

## mx.synchronize

```
synchronize(stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | None = None) -> None

Synchronize with the given stream.

Args:
  stream (Stream, optional): Stream to synchronize. If device is
     provided the default stream for that device is used. If ``None``
     then the default stream of the default device is used.
     Default: ``None``.
```

## mx.get_active_memory

```
get_active_memory() -> int

Get the actively used memory in bytes.

Note, this will not always match memory use reported by the system because
it does not include cached memory buffers.
```

## mx.get_peak_memory

```
get_peak_memory() -> int

Get the peak amount of used memory in bytes.

The maximum memory used recorded from the beginning of the program
execution or since the last call to :func:`reset_peak_memory`.
```

## mx.get_cache_memory

```
get_cache_memory() -> int

Get the cache size in bytes.

The cache includes memory not currently used that has not been returned
to the system allocator.
```

## mx.clear_cache

```
clear_cache() -> None

Clear the memory cache.

After calling this, :func:`get_cache_memory` should return ``0``.
```

## mx.set_memory_limit

```
set_memory_limit(limit: int) -> int

Set the memory limit.

The memory limit is a guideline for the maximum amount of memory to use
during graph evaluation. If the memory limit is exceeded and there is no
more RAM (including swap when available) allocations will result in an
exception.

When metal is available the memory limit defaults to 1.5 times the
maximum recommended working set size reported by the device.

Args:
  limit (int): Memory limit in bytes.

Returns:
  int: The previous memory limit in bytes.
```

## mx.set_cache_limit

```
set_cache_limit(limit: int) -> int

Set the free cache limit.

If using more than the given limit, free memory will be reclaimed
from the cache on the next allocation. To disable the cache, set
the limit to ``0``.

The cache limit defaults to the memory limit. See
:func:`set_memory_limit` for more details.

Args:
  limit (int): The cache limit in bytes.

Returns:
  int: The previous cache limit in bytes.
```

## mx.set_wired_limit

```
set_wired_limit(limit: int) -> int

Set the wired size limit.

.. note::
   * This function is only useful on macOS 15.0 or higher.
   * The wired limit should remain strictly less than the total
     memory size.

The wired limit is the total size in bytes of memory that will be kept
resident. The default value is ``0``.

Setting a wired limit larger than system wired limit is an error. You can
increase the system wired limit with:

.. code-block::

  sudo sysctl iogpu.wired_limit_mb=<size_in_megabytes>

Use :func:`device_info` to query the system wired limit
(``"max_recommended_working_set_size"``) and the total memory size
(``"memory_size"``).

Args:
  limit (int): The wired limit in bytes.

Returns:
  int: The previous wired limit in bytes.
```

## mx.device_info

```
device_info(d: mlx.core.Device | None = None) -> dict[str, str | int]

Get information about a device.

Returns a dictionary with device properties. Available keys depend
on the backend and device type. Common keys include ``device_name``,
``architecture``, and ``total_memory`` (or ``memory_size``).

Args:
    d (Device): The device to query (defaults to the default device).

Returns:
    dict: Device information.
```

## mx.metal.is_available

```
is_available() -> bool

Check if the Metal back-end is available.
```

## mx.metal.start_capture

```
start_capture(path: str) -> None

Start a Metal capture.

Args:
  path (str): The path to save the capture which should have
    the extension ``.gputrace``.
```

## mx.metal.stop_capture

```
stop_capture() -> None

Stop a Metal capture.
```

## nn.value_and_grad

```
Transform the passed function ``fn`` to a function that computes the
gradients of ``fn`` wrt the model's trainable parameters and also its
value.

Args:
    model (mlx.nn.Module): The model whose trainable parameters to compute
                           gradients for
    fn (Callable): The scalar function to compute gradients for

Returns:
    A callable that returns the value of ``fn`` and the gradients wrt the
    trainable parameters of ``model``
```

## nn.quantize

```
Quantize the sub-modules of a module according to a predicate.

By default all layers that define a ``to_quantized()`` method will be
quantized. Both :obj:`Linear` and :obj:`Embedding` layers will be
quantized. The module is updated in-place.

Note:
    ``quantize_input=True`` is only supported for ``"nvfp4"`` and ``"mxfp8"``
    modes and :obj:`Linear` layers.

Args:
    model (mlx.nn.Module): The model whose leaf modules may be quantized.
    group_size (Optional[int]): The quantization group size (see
       :func:`mlx.core.quantize`). Default: ``None``.
    bits (Optional[int]): The number of bits per parameter (see
       :func:`mlx.core.quantize`). Default: ``None``.
    mode (str): The quantization method to use (see
       :func:`mlx.core.quantize`). Default: ``"affine"``.
    quantize_input (bool): Whether to quantize activations. Default: ``False``.
    class_predicate (Optional[Callable]): A callable which receives the
       :obj:`Module` path and :obj:`Module` itself and returns ``True`` or a
       dict of params for ``to_quantized`` if it should be quantized and
       ``False`` otherwise. If ``None``, then all layers that define a
       ``to_quantized()`` method are quantized. Default: ``None``.

Example:
    Weight only quantization for all layers that define a ``to_quantized()`` method:

    >>> import mlx.nn as nn
    >>> nn.quantize(model, group_size=64, bits=4, mode="affine")

    Weight and input quantization for all linear layers:

    >>> predicate = lambda p, m: isinstance(m, nn.Linear)
    >>> nn.quantize(model, mode="nvfp4", quantize_input=True, class_predicate=predicate)
```

## nn.average_gradients

```
Average the gradients across the distributed processes in the passed group.

This helper enables concatenating several gradients of small arrays to one
big all reduce call for better networking performance.

Args:
    gradients (Any): The Python tree containing the gradients (it should
        have the same structure across processes)
    group (Optional[mlx.core.distributed.Group]): The group of processes to
        average the gradients. If set to ``None`` the global group is used.
        Default: ``None``.
    all_reduce_size (int): Group arrays until their size in bytes exceeds
        this number. Perform one communication step per group of arrays. If
        less or equal to 0 array grouping is disabled. Default: ``32MiB``.
    communication_stream (Optional[mlx.core.Stream]): The stream to use
        for the communication. If unspecified the default communication
        stream is used which can vary by back-end. Default: ``None``.
```
