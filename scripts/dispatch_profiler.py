import torch
import time
from torch.utils._python_dispatch import TorchDispatchMode

MB = 1024 * 1024.0


class OpRecord:
    def __init__(
        self,
        op_name: str,
        input_shapes: list[tuple],
        output_shapes: list[tuple],
        time_taken_on_gpu: float,
        time_taken_on_cpu: float,
        memory_taken: float,
        input_dtypes: list[torch.dtype],
        use_gpu: bool
    ):
        self.op_name = op_name
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.time_taken_on_gpu = time_taken_on_gpu
        self.time_taken_on_cpu = time_taken_on_cpu
        self.memory_taken = memory_taken
        self.input_dtypes = input_dtypes

class OpProfilerDispatchMode(TorchDispatchMode):

    # this is a dispatch mode that records the following:
    # 1. What aten op is being dispatched
    # 2. What is the input shape
    # 3. What is the output shape
    # 4. What is the time taken to dispatch the op
    # 5. What is the memory taken to dispatch the op

    def __init__(self):
        super().__init__()
        self.op_records = []

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
    #  actually dispatch the op and get the result
        use_gpu = False
        start_time = time.time()
        rs = func(*args, **kwargs)
        end_time = time.time()
        mem: float = torch.cuda.memory_allocated() / MB
        #  record the op, input shape, output shape, time taken, memory taken
        input_shapes = []
        input_dtypes = []
        
        if not torch.cuda.is_available():
            current_device = "cpu"
        else:
            current_device = torch.cuda.current_device()
        
        if "cuda" in current_device:
            cpu_start_time = time.time()
            torch.cuda.synchronize()
            cpu_end_time = time.time()
            time_taken_on_cpu = cpu_end_time - cpu_start_time
            use_gpu = True
        elif "cpu" in current_device:
            time_taken_on_gpu = 0
        else:
            raise ValueError(f"Unknown device: {current_device} right now we only support cpu and cuda")

        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_shapes.append(arg.shape)
                input_dtypes.append(arg.dtype)
            elif isinstance(arg, (int, float)):
                input_shapes.append(())  # scalar shape
                input_dtypes.append(type(arg))  # no dtype for Python scalars
            else:
                input_shapes.append(None)
                input_dtypes.append(type(arg))

        output_shapes = []
        if isinstance(rs, torch.Tensor):
            output_shapes.append(rs.shape)
        elif isinstance(rs, (int, float)):
            output_shapes.append(())  # scalar shape
        else:
            output_shapes.append(None)

        if use_gpu:
            time_taken_on_gpu = end_time - start_time
        else:
            time_taken_on_cpu = end_time - start_time

        self.op_records.append(
            OpRecord(
                op_name=func.__name__,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                time_taken_on_gpu=time_taken_on_gpu,
                time_taken_on_cpu=time_taken_on_cpu,
                memory_taken=mem,
                input_dtypes=input_dtypes,
                use_gpu=use_gpu
            )
        )
        print(f"created a record for {func.__name__}")
        return rs
    
    def get_op_records(self):
        return self.op_records



def main():
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5),
        torch.nn.Softmax(dim=1)
    )

    # Create sample input
    x = torch.randn(32, 10)

    # Enable profiling
    profiler = OpProfilerDispatchMode()
    with torch.autograd.profiler.profile() as prof:
        with profiler:
            # Run model inference
            output = model(x)

    # Print profiling results
    print("\n=== Operation Profiling Results ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get records from our custom profiler
    records = profiler.get_op_records()
    
    print("\nDetailed operation records:")
    for record in records:
        print(f"\nOperation: {record.op_name}")
        print(f"Input shapes: {record.input_shapes}")
        print(f"Output shapes: {record.output_shapes}")
        print(f"Time taken on gpu: {record.time_taken_on_gpu:.6f} seconds")
        print(f"Time taken on cpu: {record.time_taken_on_cpu:.6f} seconds")
        print(f"Memory used: {record.memory_taken:.2f} MB")

if __name__ == "__main__":
    main()