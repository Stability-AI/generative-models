import einops
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.backends.cuda import SDPBackend

from sgm.modules.attention import BasicTransformerBlock, SpatialTransformer


def benchmark_attn():
    # Lets define a helpful benchmarking function:
    # https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    # Lets define the hyper-parameters of our input
    batch_size = 32
    max_sequence_len = 1024
    num_heads = 32
    embed_dimension = 32

    dtype = torch.float16

    query = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )
    key = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )
    value = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )

    print(f"q/k/v shape:", query.shape, key.shape, value.shape)

    # Lets explore the speed of each of the 3 implementations
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # Helpful arguments mapper
    backend_map = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
    }

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    print(
        f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
    )
    with profile(
        activities=activities, record_shapes=False, profile_memory=True
    ) as prof:
        with record_function("Default detailed stats"):
            for _ in range(25):
                o = F.scaled_dot_product_attention(query, key, value)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(
        f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
    )
    with sdp_kernel(**backend_map[SDPBackend.MATH]):
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("Math implmentation stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
        try:
            print(
                f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
            )
        except RuntimeError:
            print("FlashAttention is not supported. See warnings for reasons.")
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("FlashAttention stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
        try:
            print(
                f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
            )
        except RuntimeError:
            print("EfficientAttention is not supported. See warnings for reasons.")
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("EfficientAttention stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_model(model, x, context):
    return model(x, context)


def benchmark_transformer_blocks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    checkpoint = True
    compile = False

    batch_size = 32
    h, w = 64, 64
    context_len = 77
    embed_dimension = 1024
    context_dim = 1024
    d_head = 64

    transformer_depth = 4

    n_heads = embed_dimension // d_head

    dtype = torch.float16

    model_native = SpatialTransformer(
        embed_dimension,
        n_heads,
        d_head,
        context_dim=context_dim,
        use_linear=True,
        use_checkpoint=checkpoint,
        attn_type="softmax",
        depth=transformer_depth,
        sdp_backend=SDPBackend.FLASH_ATTENTION,
    ).to(device)
    model_efficient_attn = SpatialTransformer(
        embed_dimension,
        n_heads,
        d_head,
        context_dim=context_dim,
        use_linear=True,
        depth=transformer_depth,
        use_checkpoint=checkpoint,
        attn_type="softmax-xformers",
    ).to(device)
    if not checkpoint and compile:
        print("compiling models")
        model_native = torch.compile(model_native)
        model_efficient_attn = torch.compile(model_efficient_attn)

    x = torch.rand(batch_size, embed_dimension, h, w, device=device, dtype=dtype)
    c = torch.rand(batch_size, context_len, context_dim, device=device, dtype=dtype)

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with torch.autocast("cuda"):
        print(
            f"The native model runs in {benchmark_torch_function_in_microseconds(model_native.forward, x, c):.3f} microseconds"
        )
        print(
            f"The efficientattn model runs in {benchmark_torch_function_in_microseconds(model_efficient_attn.forward, x, c):.3f} microseconds"
        )

        print(75 * "+")
        print("NATIVE")
        print(75 * "+")
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("NativeAttention stats"):
                for _ in range(25):
                    model_native(x, c)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(torch.cuda.max_memory_allocated() * 1e-9, "GB used by native block")

        print(75 * "+")
        print("Xformers")
        print(75 * "+")
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("xformers stats"):
                for _ in range(25):
                    model_efficient_attn(x, c)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(torch.cuda.max_memory_allocated() * 1e-9, "GB used by xformers block")


def test01():
    # conv1x1 vs linear
    from sgm.util import count_params

    conv = torch.nn.Conv2d(3, 32, kernel_size=1).cuda()
    print(count_params(conv))
    linear = torch.nn.Linear(3, 32).cuda()
    print(count_params(linear))

    print(conv.weight.shape)

    # use same initialization
    linear.weight = torch.nn.Parameter(conv.weight.squeeze(-1).squeeze(-1))
    linear.bias = torch.nn.Parameter(conv.bias)

    print(linear.weight.shape)

    x = torch.randn(11, 3, 64, 64).cuda()

    xr = einops.rearrange(x, "b c h w -> b (h w) c").contiguous()
    print(xr.shape)
    out_linear = linear(xr)
    print(out_linear.mean(), out_linear.shape)

    out_conv = conv(x)
    print(out_conv.mean(), out_conv.shape)
    print("done with test01.\n")


def test02():
    # try cosine flash attention
    import time

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("testing cosine flash attention...")
    DIM = 1024
    SEQLEN = 4096
    BS = 16

    print(" softmax (vanilla) first...")
    model = BasicTransformerBlock(
        dim=DIM,
        n_heads=16,
        d_head=64,
        dropout=0.0,
        context_dim=None,
        attn_mode="softmax",
    ).cuda()
    try:
        x = torch.randn(BS, SEQLEN, DIM).cuda()
        tic = time.time()
        y = model(x)
        toc = time.time()
        print(y.shape, toc - tic)
    except RuntimeError as e:
        # likely oom
        print(str(e))

    print("\n now flash-cosine...")
    model = BasicTransformerBlock(
        dim=DIM,
        n_heads=16,
        d_head=64,
        dropout=0.0,
        context_dim=None,
        attn_mode="flash-cosine",
    ).cuda()
    x = torch.randn(BS, SEQLEN, DIM).cuda()
    tic = time.time()
    y = model(x)
    toc = time.time()
    print(y.shape, toc - tic)
    print("done with test02.\n")


if __name__ == "__main__":
    # test01()
    # test02()
    # test03()

    # benchmark_attn()
    benchmark_transformer_blocks()

    print("done.")
